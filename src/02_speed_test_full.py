import argparse
import time
import ir_measures
import numpy as np
import pandas as pd
import transformers
import ir_datasets
import asnq_dataset
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
from typing import List
from tqdm import tqdm
from ir_measures import ScoredDoc, Qrel

from eemb import BertConfig, BertTokenizer
from eemb.modeling_highway_bert import BertForSequenceClassification
from utils import simple_batching, PerfRow, FIXED_EVALUATION_METRIC, print_stats, add_common_parsing_args, \
    get_scoreddocs


def test_encoder_decoder(device: torch.device,
                         docstore,
                         model_size,
                         model: transformers.BertForSequenceClassification,
                         query_store: dict,
                         scoreddoc_dict: dict,
                         tokenizer: transformers.BertTokenizerFast,
                         batch_size: int,
                         qrels: List[Qrel],
                         ir_measures_metric: ir_measures.Measure,
                         test_id: int,
                         score_token_id: int = 32089,
                         truncate_query: bool = False
                         ):
    q_ids_list = list(scoreddoc_dict.keys())

    model = model.to(device)
    model = model.eval()

    stats = {}
    run = []
    start_perf_counter = time.perf_counter_ns()

    for qid in tqdm(q_ids_list):

        tokenized_batch_list = []
        docs_batch_ids = []
        scoreddoc_per_qid = scoreddoc_dict[qid]

        for batch_scoreddoc_per_qid in simple_batching(scoreddoc_per_qid, batch_size):
            query_text = query_store[qid]
            if truncate_query and len(query_text.split()) > 32:
                query_text = " ".join(query_text.split()[:32])
            batch_queries = [query_text] * len(batch_scoreddoc_per_qid)
            batch_queries_docs = [docstore.get(doc_id).text for doc_id in batch_scoreddoc_per_qid]

            batch_input = [
                f"Query: {query_text} Document: {doc_text}" for doc_text in batch_queries_docs
            ]

            batch_tokenized = tokenizer(batch_input, return_tensors="pt", padding="max_length", truncation=True)
            tokenized_batch_list.append(batch_tokenized)
            docs_batch_ids.append([doc_id for doc_id in batch_scoreddoc_per_qid])

        scoreddoc_counter = 0
        time_elapsed = 0
        for batch_n, tokenized in enumerate(tokenized_batch_list):
            et_query_start = time.perf_counter_ns()
            if type(tokenized) == dict:
                tokenized = {k: v.to(device) for k, v in tokenized.items()}
            else:
                tokenized = tokenized.to(device)

            decoder_ids = torch.full((tokenized['input_ids'].shape[0], 1), tokenizer.pad_token_id,
                                     dtype=torch.long).to(device)
            outputs = model.forward(**tokenized, decoder_input_ids=decoder_ids)

            scoreddoc_counter += len(outputs["logits"])

            logits = outputs["logits"]
            scores = (logits[:, -1, :])[:, score_token_id]

            torch.cuda.synchronize()
            time_elapsed += time.perf_counter_ns() - et_query_start

            for i, score in enumerate(scores):
                score = score.item()
                query_id = qid
                doc_id = docs_batch_ids[batch_n][i]
                run.append(ScoredDoc(query_id=query_id, doc_id=doc_id, score=score))

        stats[qid] = {"elapsed_time": time_elapsed,
                      "scoreddoc_number": scoreddoc_counter,
                      "estimated_cost": scoreddoc_counter * model_size * 2}

    torch.cuda.synchronize()
    elapsed_perf_counter = time.perf_counter_ns() - start_perf_counter

    total_scoreddoc_number = sum([stats[qid]["scoreddoc_number"] for qid in q_ids_list])
    sum_of_query_times = sum([stats[qid]["elapsed_time"] for qid in q_ids_list])

    print_stats(elapsed_perf_counter, sum_of_query_times, total_scoreddoc_number)
    # Compute the effectiveness
    ndcg_per_query_exit = list(ir_measures_metric.iter_calc(qrels, run))

    # Merge the efficiency values with the effectiveness
    res = []
    # Hardcoded for now
    for query_id, metric, value in ndcg_per_query_exit:
        res.append(PerfRow(test_id=test_id,
                           query_id=query_id,
                           estimated_cost=stats[query_id]["estimated_cost"],
                           query_time=stats[query_id]["elapsed_time"],
                           scoreddoc_counter=stats[query_id]["scoreddoc_number"],
                           metric=str(metric),
                           value=value))
    return res


# Test without early exit
def test_encoder_model(device: torch.device,
                       docstore,
                       model_size,
                       model: transformers.BertForSequenceClassification,
                       query_store: dict,
                       scoreddoc_dict: dict,
                       tokenizer: transformers.BertTokenizerFast,
                       batch_size: int,
                       qrels: List[Qrel],
                       ir_measures_metric: ir_measures.Measure,
                       test_id: int,
                       score_token_id: int = 32089,
                       truncate_query: bool = False,
                       ):
    q_ids_list = list(scoreddoc_dict.keys())

    model = model.to(device)
    model = model.eval()

    stats = {}
    run = []
    start_perf_counter = time.perf_counter_ns()

    for qid in tqdm(q_ids_list):

        tokenized_batch_list = []
        docs_batch_ids = []
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
            docs_batch_ids.append([doc_id for doc_id in batch_scoreddoc_per_qid])

        scoreddoc_counter = 0
        time_elapsed = 0
        for batch_n, tokenized in enumerate(tokenized_batch_list):
            et_query_start = time.perf_counter_ns()
            if type(tokenized) == dict:
                tokenized = {k: v.to(device) for k, v in tokenized.items()}
            else:
                tokenized = tokenized.to(device)

            outputs = model(**tokenized)

            scoreddoc_counter += len(outputs["logits"])

            logits = outputs["logits"]

            if logits.shape[1] == 1:  # cross-encoder/ms-marco-MiniLM-L-12-v2 outputs only one values (relevance)
                scores = torch.sigmoid(logits).ravel()
            else:
                scores = torch.softmax(logits, dim=1)[:, 1]

            torch.cuda.synchronize()
            time_elapsed += time.perf_counter_ns() - et_query_start

            for i, score in enumerate(scores):
                score = score.item()
                query_id = qid
                doc_id = docs_batch_ids[batch_n][i]
                run.append(ScoredDoc(query_id=query_id, doc_id=doc_id, score=score))

        stats[qid] = {"elapsed_time": time_elapsed,
                      "scoreddoc_number": scoreddoc_counter,
                      "estimated_cost": scoreddoc_counter * model_size}

    torch.cuda.synchronize()
    elapsed_perf_counter = time.perf_counter_ns() - start_perf_counter

    total_scoreddoc_number = sum([stats[qid]["scoreddoc_number"] for qid in q_ids_list])
    sum_of_query_times = sum([stats[qid]["elapsed_time"] for qid in q_ids_list])

    print_stats(elapsed_perf_counter, sum_of_query_times, total_scoreddoc_number)

    # Compute the effectiveness
    ndcg_per_query_exit = list(ir_measures_metric.iter_calc(qrels, run))

    # Merge the efficiency values with the effectiveness
    res = []
    # Hardcoded for now
    for query_id, metric, value in ndcg_per_query_exit:
        res.append(PerfRow(test_id=test_id,
                           query_id=query_id,
                           estimated_cost=stats[query_id]["estimated_cost"],
                           query_time=stats[query_id]["elapsed_time"],
                           scoreddoc_counter=stats[query_id]["scoreddoc_number"],
                           metric=str(metric),
                           value=value))
    return res


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(prog="Performance evaluation",
                                     description="Evaluate the performance of a model on a dataset.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_family", type=str, default='cls')
    add_common_parsing_args(parser)

    args = parser.parse_args()

    model_name = args.model
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
    model_family = args.model_family

    device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
    if model_path is not None and model_name is not None:
        raise ValueError("Please specify either model or model_path, not both.")

    if model_path is None:  # we are using a standard model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, output_hidden_states=True)
        model = model.to(device)

        model.eval()
    else:  # we are using the original eemb model
        model_path = Path(model_path)
        config = BertConfig.from_pretrained(model_path / "config.json", num_labels=2)
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
        model = BertForSequenceClassification.from_pretrained(model_path, config=config)
        model.core.encoder.set_early_exit_thresholds(args)
        model = model.to(device)

        model = model.eval()

    dataset = ir_datasets.load(dataset_name)
    docstore = dataset.docs_store()
    query_store = {query.query_id: query.text for query in dataset.queries_iter()}

    scoreddoc_dict = get_scoreddocs(dataset, dataset_name, scoreddoc_folder, list_truncation)

    print("Total number of scoreddoc elem: ", sum([len(v) for v in scoreddoc_dict.values()]))

    qrels = list(dataset.qrels_iter())

    res = []

    # TODO hardcoded for this poc, to do something better in future work
    # truncate_query = model_name in ["veneres/monoelectra",
    #                                 "cross-encoder/ms-marco-MiniLM-L-12-v2",
    #                                 "naver/trecdl22-crossencoder-electra"]
    # truncate_query = truncate_query and dataset_name == "beir/arguana"
    # if truncate_query:
    #     print("Truncating the queries to 32 tokens")
    truncate_query = False
    print(f"Model is : {model_family}")
    with torch.no_grad():
        for i in range(args.n_reps):
            if model_family == 'cls':
                partial_res = test_encoder_model(device=device,
                                                 docstore=docstore,
                                                 model=model,
                                                 # TODO hardcoding the model size for now
                                                 model_size=24 if model_name is not None and "naver" in model_name else 12,
                                                 query_store=query_store,
                                                 scoreddoc_dict=scoreddoc_dict,
                                                 tokenizer=tokenizer,
                                                 batch_size=batch_size,
                                                 qrels=qrels,
                                                 ir_measures_metric=eval_metric,
                                                 test_id=i,
                                                 truncate_query=truncate_query)
            elif model_family == 'seq':
                partial_res = test_encoder_decoder(device=device,
                                                   docstore=docstore,
                                                   model_size=12,
                                                   model=model,
                                                   query_store=query_store,
                                                   scoreddoc_dict=scoreddoc_dict,
                                                   tokenizer=tokenizer,
                                                   batch_size=batch_size,
                                                   qrels=qrels,
                                                   ir_measures_metric=eval_metric,
                                                   test_id=i,
                                                   truncate_query=truncate_query)
            else:
                raise ValueError(f"model family: {model_family} is not supported")
            res.extend(partial_res)
            pd.DataFrame(res).to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
