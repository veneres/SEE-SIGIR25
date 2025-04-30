import numpy as np
import pandas as pd
import transformers
import argparse
import time
import ir_datasets
import asnq_dataset
import torch
import gc

from pathlib import Path
from typing import List, Any
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ir_measures import ScoredDoc, Qrel

from mini_berts import AutoModelFirstSliced, AutoModelLastSliced
from utils import apply_ee_torch, EarlyExitMode, add_common_parsing_args, get_scoreddocs, PADDING_MEMORY, \
    simple_batching, NEGATIVE_CONTRIB_EE, PerfRow, FIXED_EVALUATION_METRIC, print_stats

import pdb


def test_ee(device: torch.device,
            docstore,
            model,
            query_store: dict,
            scoreddoc_dict: dict,
            tokenizer: Any,
            batch_size: int,
            exit_level: int,
            exit_mode: EarlyExitMode,
            exit_param: float,
            qrels: List[Qrel],
            ir_measures_metric,
            test_id: int,
            model_name=None,
            exit_k: int = 10,
            exit_sim="amax",
            truncate_query=False
            ):

    model_first_part = AutoModelFirstSliced.from_pretrained(model, tokenizer, end_layer=exit_level,
                                                            sim_function=exit_sim)
    model_second_part = AutoModelLastSliced.from_pretrained(model, start_layer=exit_level)

    model_first_part.to(model.device)
    model_first_part = model_first_part.eval()

    model_second_part.to(model.device)
    model_second_part = model_second_part.eval()
    q_ids_list = list(map(str, scoreddoc_dict.keys()))

    run = []

    memory_buffer_size = max([len(scoreddoc_dict[qid]) for qid in q_ids_list]) + PADDING_MEMORY

    ee_values = torch.zeros(memory_buffer_size).to(device)
    scores = torch.zeros(memory_buffer_size).to(device)
    if "naver" in model_name:
        hidden_states = torch.zeros((memory_buffer_size, 512, 1024)).to(device)
        attention_masks = torch.zeros((memory_buffer_size, 1, 1, 512)).to(device)
    elif "MiniLM-L-12-v2" in model_name:
        hidden_states = torch.zeros((memory_buffer_size, 512, 384)).to(device)
        attention_masks = torch.zeros((memory_buffer_size, 1, 1, 512)).to(device)
    else:
        hidden_states = torch.zeros((memory_buffer_size, 512, 768)).to(device)
        attention_masks = torch.zeros((memory_buffer_size, 1, 1, 512)).to(device)

    stats = {}

    # End preamble start test

    et_total_start = time.perf_counter_ns()

    for qid in tqdm(q_ids_list):

        tokenized_batch_list = []
        scoreddoc_per_qid = scoreddoc_dict[qid]

        for batch_scoreddoc_per_qid in simple_batching(scoreddoc_per_qid, batch_size):
            query_text = query_store[qid]
            if truncate_query and len(query_text.split()) > 32:
                query_text = " ".join(query_text.split()[:32])

            batch_queries = [query_text] * len(batch_scoreddoc_per_qid)
            batch_queries_docs = [docstore.get(doc_id).text for doc_id in batch_scoreddoc_per_qid]

            tokenized_batch = tokenizer(batch_queries,
                                        batch_queries_docs,
                                        return_tensors="pt",
                                        padding="max_length",
                                        truncation=True)
            tokenized_batch_list.append(tokenized_batch)

        et_query_start = time.perf_counter_ns()
        # First slice start

        n_docs_per_query = len(scoreddoc_dict[qid])

        for batch_n, tokenized in enumerate(tokenized_batch_list):
            tokenized = tokenized.to(device)
            outputs = model_first_part(**tokenized)

            start_slicing = batch_n * batch_size
            end_slicing = min((batch_n + 1) * batch_size, n_docs_per_query)

            ee_values[start_slicing:end_slicing] = outputs["similarities"]
            hidden_states[start_slicing:end_slicing] = outputs["hidden_states"]
            attention_masks[start_slicing:end_slicing] = outputs["attention_mask"]

        # First slice end

        # EE start
        not_exit_indices = torch.zeros(n_docs_per_query, dtype=torch.bool)
        min_value = torch.min(ee_values[:n_docs_per_query])
        max_value = torch.max(ee_values[:n_docs_per_query])
        scaled_values = (ee_values[: n_docs_per_query] - min_value) / (max_value - min_value + 1e-8)
        ee_values[: n_docs_per_query] = scaled_values
        not_exit_indices[: n_docs_per_query] = apply_ee_torch(scaled_values, exit_mode, exit_param, exit_k)

        exited = torch.logical_not(not_exit_indices).sum().item()
        passed = not_exit_indices.sum().item()

        ee_index = torch.nonzero(torch.logical_not(not_exit_indices))
        scores[ee_index] = ee_values[ee_index] - NEGATIVE_CONTRIB_EE

        # EE end

        last_slice_iterator = enumerate(
            range(0, len(hidden_states[:n_docs_per_query, :][not_exit_indices]), batch_size))

        # Last slice start
        for batch_n, batch_start in last_slice_iterator:
            batch_end = min(batch_start + batch_size, len(hidden_states[:n_docs_per_query, :][not_exit_indices]))

            outputs = model_second_part(
                hidden_states[:n_docs_per_query, :][not_exit_indices][batch_start:batch_end],
                attention_masks[:n_docs_per_query, :][not_exit_indices][batch_start:batch_end])

            index_to_update = torch.nonzero(not_exit_indices)[batch_start: batch_end]
            scores[index_to_update] = outputs['scores'].reshape(-1, 1)

        # Last slice end

        torch.cuda.synchronize()
        query_elapsed_time = time.perf_counter_ns() - et_query_start
        for elem in range(n_docs_per_query):
            query_id = qid
            doc_id = scoreddoc_dict[qid][elem]
            score = scores[elem].item()
            run.append(ScoredDoc(query_id=query_id, doc_id=doc_id, score=score))
        stats[qid] = {"elapsed_time": query_elapsed_time,
                      "exited": exited,
                      "passed": passed}  # We left the computation of the effectiveness in the end

    torch.cuda.synchronize()
    total_elapsed_real = time.perf_counter_ns() - et_total_start

    total_scoreddoc_number = sum([qid_stats["exited"] + qid_stats["passed"] for qid, qid_stats in stats.items()])
    sum_of_query_times = sum([stats[qid]["elapsed_time"] for qid in q_ids_list])

    print_stats(total_elapsed_real, sum_of_query_times, total_scoreddoc_number)

    # Compute the effectiveness
    ndcg_per_query_exit = list(ir_measures_metric.iter_calc(qrels, run))

    # Merge the efficiency values with the effectiveness
    res = []
    # Hardcoded for now
    n_transformer_block = 12 if not "naver" in model_name else 24
    for query_id, metric, value in ndcg_per_query_exit:
        res.append(PerfRow(test_id=test_id,
                           query_id=query_id,
                           estimated_cost=stats[query_id]["exited"] * exit_level + stats[query_id][
                               "passed"] * n_transformer_block,
                           query_time=stats[query_id]["elapsed_time"],
                           scoreddoc_counter=stats[query_id]["exited"] + stats[query_id]["passed"],
                           metric=str(metric),
                           value=value))

    torch.cuda.empty_cache()
    gc.collect()

    return res


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(prog="Performance evaluation",
                                     description="Evaluate the performance of a model on a dataset.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--exit_level", type=int, default=0)
    parser.add_argument("--exit_mode",
                        type=int,
                        default=1,
                        help="1: static, 2: proximity static, 3: rank, 3: top_percent, 4: proximity_percent")
    parser.add_argument("--exit_param",
                        type=float,
                        default=0.5,
                        help="threshold for static, rank, proximity_static, k for proximity_percent")
    parser.add_argument("--exit_k", type=int, default=10, help="k for top-k")
    parser.add_argument("--exit_sim", type=str, default="amax",
                        help="Exit similarity function (amax, max, mean, mean_centroid)")

    add_common_parsing_args(parser)

    args = parser.parse_args()

    model_name = args.model
    device = args.device
    dataset_name = args.dataset
    batch_size = args.batch_size
    exit_mode = EarlyExitMode(args.exit_mode)
    exit_param = args.exit_param
    exit_k = args.exit_k
    output_file = args.output_file
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    exit_level = args.exit_level
    eval_metric = args.eval_metric
    eval_metric = FIXED_EVALUATION_METRIC[eval_metric]
    scoreddoc_folder = Path(args.scoreddoc_path)

    list_truncation = args.list_truncation
    model_family = args.model_family

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, output_hidden_states=True)
    model = model.to(device)

    model.eval()

    dataset = ir_datasets.load(dataset_name)
    docstore = dataset.docs_store()
    query_store = {query.query_id: query.text for query in dataset.queries_iter()}

    scoreddoc_dict = get_scoreddocs(dataset, dataset_name, scoreddoc_folder, list_truncation)

    print("Total number of scoreddoc elem: ", sum([len(v) for v in scoreddoc_dict.values()]))

    qrels = list(dataset.qrels_iter())

    # TODO hardcoded for this poc, to do something better in future work
    truncate_query = model_name in ["veneres/monoelectra",
                                    "cross-encoder/ms-marco-MiniLM-L-12-v2",
                                    "naver/trecdl22-crossencoder-electra"]
    truncate_query = truncate_query and dataset_name == "beir/arguana"
    if truncate_query:
        print("Truncating the queries to 32 tokens")

    res = []
    with torch.no_grad():
        for i in range(args.n_reps):
            partial_res = test_ee(device=device,
                                  docstore=docstore,
                                  model=model,
                                  query_store=query_store,
                                  scoreddoc_dict=scoreddoc_dict,
                                  tokenizer=tokenizer,
                                  batch_size=batch_size,
                                  exit_level=exit_level,
                                  exit_mode=exit_mode,
                                  exit_param=exit_param,
                                  exit_k=exit_k,
                                  exit_sim=args.exit_sim,
                                  qrels=qrels,
                                  model_name=model_name,
                                  ir_measures_metric=eval_metric,
                                  test_id=i,
                                  truncate_query=truncate_query,
                                  model_family=model_family)
            res.extend(partial_res)
            pd.DataFrame(res).to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
