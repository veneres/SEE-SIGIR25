import argparse
import time
import ir_measures
import numpy as np
import pandas as pd
import ir_datasets
import asnq_dataset
import torch
import gc

from tqdm import tqdm
from pathlib import Path
from typing import List
from ir_measures import ScoredDoc, Qrel

from mini_berts import EEMBSlice
from utils import print_stats, simple_batching, FIXED_EVALUATION_METRIC, PerfRow, add_common_parsing_args, \
    get_scoreddocs, PADDING_MEMORY

def test_eemb(device: torch.device,
              docstore,
              model_path: Path,
              query_store: dict,
              scoreddoc_dict: dict,
              batch_size: int,
              pos_conf: float,
              neg_conf: float,
              qrels: List[Qrel],
              ir_measures_metric: ir_measures.Measure,
              test_id: int,
              truncate_query: bool = False
              ):
    eeMB_slices, tokenizer = EEMBSlice.from_pretrained(model_path, device)

    eeMB_slices = [slice.to(device) for slice in eeMB_slices]
    eeMB_slices = [slice.eval() for slice in eeMB_slices]

    run = []

    q_ids_list = list(scoreddoc_dict.keys())

    memory_buffer_size = max([len(scoreddoc_dict[qid]) for qid in q_ids_list]) + PADDING_MEMORY

    ee_values = torch.zeros(memory_buffer_size).to(device)
    hidden_states = torch.zeros((memory_buffer_size, 512, 768)).to(device)
    attention_masks = torch.zeros((memory_buffer_size, 1, 1, 512)).to(device)

    stats = {}

    # End preamble start test

    et_total_start = time.perf_counter_ns()

    for qid in tqdm(q_ids_list):

        estimated_cost = 0

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

        not_exit_indices = None

        batching = enumerate(tokenized_batch_list)

        for slice_n, slice in enumerate(eeMB_slices):

            for batch_n, batch in batching:
                if slice_n == 0:
                    batch = batch.to(device)  # first slice, batch is the tokenized batch
                    embeddings, attention_mask, probs = slice(**batch)
                    start_slicing = batch_n * batch_size
                    end_slicing = min((batch_n + 1) * batch_size, n_docs_per_query)

                    estimated_cost += end_slicing - start_slicing

                    ee_values[start_slicing:end_slicing] = probs[:, 1]
                    hidden_states[start_slicing:end_slicing] = embeddings
                    attention_masks[start_slicing:end_slicing] = attention_mask
                else:
                    batch_start = batch  # after the first slice, batch is the start index of the batch

                    batch_end = min(batch + batch_size, len(hidden_states[:n_docs_per_query, :][not_exit_indices]))

                    estimated_cost += batch_end - batch_start

                    embeddings = hidden_states[:n_docs_per_query][not_exit_indices][batch_start:batch_end]
                    attention_mask = attention_masks[:n_docs_per_query, :][not_exit_indices][batch_start:batch_end]
                    embeddings, attention_mask, probs = slice(embeddings=embeddings, attention_mask=attention_mask)
                    rows_to_update = torch.nonzero(not_exit_indices).squeeze(1)[batch_start:batch_end]
                    ee_values[rows_to_update] = probs[:, 1]
                    hidden_states[rows_to_update, :] = embeddings
                    attention_masks[rows_to_update, :] = attention_mask

            # EE start
            not_exit_indices = torch.logical_and(ee_values[:n_docs_per_query] < pos_conf,
                                                 1 - ee_values[:n_docs_per_query] < neg_conf)

            not_exit_indices.sum()

            if not_exit_indices.sum() == 0:
                break

            batching = enumerate(
                range(0, len(hidden_states[:n_docs_per_query, :][not_exit_indices]), batch_size))

        # Last slice end
        torch.cuda.synchronize()
        stats[qid] = {"elapsed_time": time.perf_counter_ns() - et_query_start,
                      "estimated_cost": estimated_cost,
                      "scoreddoc_number": n_docs_per_query}  # We left the computation of the effectiveness in the end

        for i, score in enumerate(ee_values[:n_docs_per_query]):
            score = score.item()
            query_id = qid
            doc_id = scoreddoc_dict[qid][i]

            run.append(ScoredDoc(query_id=query_id, doc_id=doc_id, score=score))

    torch.cuda.synchronize()
    total_elapsed_real = time.perf_counter_ns() - et_total_start

    total_scoreddoc_number = sum([stats[qid]["scoreddoc_number"] for qid in q_ids_list])
    sum_of_query_times = sum([stats[qid]["elapsed_time"] for qid in q_ids_list])

    print_stats(total_elapsed_real, sum_of_query_times, total_scoreddoc_number)

    # Compute the effectiveness
    ndcg_per_query_exit = list(ir_measures_metric.iter_calc(qrels, run))

    # Merge the efficiency values with the effectiveness
    res = []

    for query_id, metric, value in ndcg_per_query_exit:
        res.append(PerfRow(test_id=test_id,
                           query_id=query_id,
                           estimated_cost=stats[query_id]["estimated_cost"],
                           query_time=stats[query_id]["elapsed_time"],
                           scoreddoc_counter=stats[query_id]["scoreddoc_number"],
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
    parser.add_argument("--model_path", type=str, default="/data/models/eemb/msmarco/")
    parser.add_argument("--eemb_original", default=False, action="store_true")
    parser.add_argument("--pc", default=1.0, type=float,
                        help="Positive confidence threshold for early exit.")
    parser.add_argument("--nc", default=0.95, type=float,
                        help="Negative confidence threshold for early exit.")

    add_common_parsing_args(parser)

    args = parser.parse_args()

    model_path = args.model_path
    device = args.device
    dataset_name = args.dataset
    batch_size = args.batch_size
    output_file = args.output_file
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    eval_metric = args.eval_metric
    eval_metric = FIXED_EVALUATION_METRIC[eval_metric]
    scoreddoc_folder = Path(args.scoreddoc_path)
    list_truncation = args.list_truncation

    dataset = ir_datasets.load(dataset_name)
    docstore = dataset.docs_store()
    query_store = {query.query_id: query.text for query in dataset.queries_iter()}

    scoreddoc_dict = get_scoreddocs(dataset, dataset_name, scoreddoc_folder, list_truncation)

    print("Total number of scoreddoc elem: ", sum([len(v) for v in scoreddoc_dict.values()]))

    qrels = list(dataset.qrels_iter())

    truncate_query = dataset_name == "beir/arguana"
    if truncate_query:
        print("Truncating the queries to 32 tokens")


    res = []
    with torch.no_grad():
        for i in range(args.n_reps):
            partial_res = test_eemb(device=device,
                                    docstore=docstore,
                                    model_path=model_path,
                                    query_store=query_store,
                                    scoreddoc_dict=scoreddoc_dict,
                                    batch_size=batch_size,
                                    pos_conf=args.pc,
                                    neg_conf=args.nc,
                                    qrels=qrels,
                                    ir_measures_metric=eval_metric,
                                    test_id=i,
                                    truncate_query=truncate_query)

            res.extend(partial_res)
            pd.DataFrame(res).to_csv(output_file, index=False)





if __name__ == '__main__':
    main()
