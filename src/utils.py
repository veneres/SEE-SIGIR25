import argparse
from enum import Enum
from pathlib import Path

import ir_datasets
import numpy as np
import pandas as pd
import torch
from itertools import islice
import ir_measures
from collections import namedtuple

from collections import defaultdict
from tqdm import tqdm

PADDING_MEMORY = 10  # just to be sure in the poc

NEGATIVE_CONTRIB_EE = 1000  # to put ee values in the negative range

PerfRow = namedtuple("PerfRow",
                     ["test_id", "query_id", "estimated_cost", "query_time", "scoreddoc_counter", "metric", "value"])

FIXED_EVALUATION_METRIC = {
    "ndcg": ir_measures.nDCG @ 10,
    "rr": ir_measures.RR @ 10,
}


def simple_batching(iterable, batch_size=64):
    batch = []
    for i in iterable:
        batch.append(i)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def print_stats(elapsed_perf_counter: int, sum_of_query_times: float, total_scoreddoc_number: int) -> None:
    print(f"ScoredDoc number: {total_scoreddoc_number}")
    print(f"Total time: {elapsed_perf_counter / 1e6} ms")
    print(f"Time per doc including non-essential operations: {elapsed_perf_counter / total_scoreddoc_number / 1e6} ms")
    print(f"Time per doc to be used: {sum_of_query_times / total_scoreddoc_number / 1e6} ms")

def add_common_parsing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="msmarco-passage/trec-dl-2019/judged")
    parser.add_argument("--output_file", type=str, default="speed_stats.csv")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_reps", type=int, default=10)
    parser.add_argument("--scoreddoc_path", default="/data/beir-bm25-runs", type=str)
    parser.add_argument("--eval_metric", default="ndcg", type=str)
    parser.add_argument("--list_truncation", type=int, default=None)

def get_scoreddocs(dataset: ir_datasets.Dataset, dataset_name: str, scoreddoc_folder: Path, list_truncation: int) -> dict:
    scoreddoc_dict = defaultdict(list)
    scoreddoc_dict_with_scores = defaultdict(list)
    if scoreddoc_folder is None:
        for query_id, doc_id, score in dataset.scoreddocs_iter():
            scoreddoc_dict[query_id].append(doc_id)
        if list_truncation is not None:
            raise ValueError("list_truncation should be None if scoreddoc_folder is not None")
    else:
        custom_scoreddoc = scoreddoc_folder / f"{dataset_name.replace('/', '_')}.csv"
        custom_scoreddoc = pd.read_csv(custom_scoreddoc, dtype={"query_id": str, "doc_id": str, "score": float})

        for elem_idx, elem in tqdm(custom_scoreddoc.iterrows(),
                                   total=len(custom_scoreddoc),
                                   desc="Loading custom scoreddoc",
                                   leave=False):
            query_id = elem["query_id"]
            doc_id = elem["doc_id"]
            score = elem["score"]
            if type(doc_id) != str:
                print("Warning: problematic doc_id, skipping for the moment")
                print(elem_idx)
                print(elem)
                continue
            scoreddoc_dict_with_scores[query_id].append((score, doc_id))

        for query_id, doc_id_and_score_list in scoreddoc_dict_with_scores.items():
            doc_id_and_score_list.sort(reverse=True)
            doc_id_and_score_list = doc_id_and_score_list[:list_truncation]
            scoreddoc_dict[query_id] = [doc_id for _, doc_id in doc_id_and_score_list]

    return scoreddoc_dict



class EarlyExitMode(Enum):
    STATIC_THRESHOLD = 1
    PROXIMITY_STATIC = 2
    RANK = 3
    TOP_PERCENT = 4
    PROXIMITY_PERCENT = 5


class SimilarityType(Enum):
    MAXSIM = 1
    AVGSIM = 2
    MINSIM = 3

    MAXOVR = 4
    AVGOVR = 5
    MINOVR = 6


def apply_EE(ee_values, mode, param, k=10):
    if mode == EarlyExitMode.STATIC_THRESHOLD:
        return ee_values >= param
    elif mode == EarlyExitMode.PROXIMITY_STATIC:
        top_values = (np.sort(ee_values)[::-1])[:k]
        lts = max(0, np.min(top_values) - param) if top_values.size > 0 else 0.0
        return ee_values >= lts
    elif mode == EarlyExitMode.PROXIMITY_PERCENT:
        p = 1 - param
        sorted_values = (np.sort(ee_values)[::-1])
        tail_values = sorted_values[k:]
        percentile = np.percentile(tail_values, 100 * p) if tail_values.size > 0 else 0.0
        lts = percentile
        return ee_values >= lts  # at k values are kept
    elif mode == EarlyExitMode.TOP_PERCENT:
        # compute p-th percentile of ee values
        p = 1 - param
        percentile = np.percentile(ee_values, 100 * p) if ee_values.size > 0 else 0.0
        lts = max(0, percentile)
        return ee_values >= lts
    elif mode == EarlyExitMode.RANK:
        p = max(k, int(param))
        top_values = (np.sort(ee_values)[::-1])[:p]
        lts = max(0, np.min(top_values)) if top_values.size > 0 else 0.0
        return ee_values >= lts
    else:
        return ee_values >= 0


def apply_ee_torch(ee_values: torch.Tensor, mode: EarlyExitMode, param: float, k=10):
    if mode == EarlyExitMode.STATIC_THRESHOLD:
        return ee_values >= param
    elif mode == EarlyExitMode.PROXIMITY_STATIC:
        top_values = torch.sort(ee_values, descending=True)[0][:k]
        lts = max(0, torch.min(top_values).item() - param) if len(top_values) > 0 else 0.0
        return ee_values >= lts
    elif mode == EarlyExitMode.PROXIMITY_PERCENT:
        p = 1 - param
        sorted_values = torch.sort(ee_values, descending=True)[0]
        tail_values = sorted_values[k:]
        percentile = torch.quantile(tail_values, p) if len(tail_values) > 0 else 0.0
        lts = percentile
        return ee_values >= lts  # at k values are kept
    elif mode == EarlyExitMode.TOP_PERCENT:
        # compute p-th percentile of ee values
        p = 1 - param
        percentile = torch.quantile(ee_values, p) if len(ee_values) > 0 else torch.tensor(0.0)
        lts = max(0, percentile.item())
        return ee_values >= lts
    elif mode == EarlyExitMode.RANK:
        p = k + int(param * 100)
        top_values = torch.sort(ee_values, descending=True)[0][:p]
        lts = max(0, torch.min(top_values).item()) if len(top_values) > 0 else 0.0
        return ee_values >= lts
    else:
        return ee_values >= 0


def get_valid_similarity_names():
    return [s.name.lower() for s in SimilarityType]


def batched_iterable(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch
