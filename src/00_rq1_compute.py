import argparse
import logging
import math
from pathlib import Path
from typing import Tuple, Dict, List, Any

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import ir_datasets
import asnq_dataset
from tqdm import tqdm
import torch
from ir_measures import ScoredDoc

from mini_berts import AutoModelFirstSliced, AutoModelLastSliced
from utils import batched_iterable

FIXED_SIM_TYPES = ["max", "amax", "mean", "mean_centroid"]


# Returning values in the form:
# {"sym_type": {
#              "layer_num": [ScoredDoc, ScoredDoc, ...],}},
# [ScoredDoc, ScoredDoc, ...]
def compute_first_slice_outputs(model_name: str,
                                dataset: ir_datasets.Dataset,
                                scoreddoc: pd.DataFrame,
                                tokenizer,
                                exit_level: str,
                                batch_size: int,
                                device: str) -> Tuple[Dict[str, Dict[int, List[Any]]], List[ScoredDoc]]:
    original_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    original_model = original_model.to(device)

    exit_level = int(exit_level)
    first_slice_model = AutoModelFirstSliced.from_pretrained(model=original_model,
                                                             tokenizer=tokenizer,
                                                             end_layer=exit_level,
                                                             sim_function=None)
    first_slice_model = first_slice_model.to(device)
    first_slice_model = first_slice_model.eval()

    last_slice_model = AutoModelLastSliced.from_pretrained(model=original_model,
                                                           start_layer=exit_level)
    last_slice_model = last_slice_model.to(device)
    last_slice_model = last_slice_model.eval()

    scoreddoc_bm25 = []
    for _, elem in scoreddoc.iterrows():
        scoreddoc_bm25.append(ScoredDoc(query_id=elem["query_id"], doc_id=elem["doc_id"], score=elem["score"]))
    queries_subset = set(scoreddoc["query_id"].unique())

    scoreddoc_bm25_sorted = sorted(scoreddoc_bm25, key=lambda x: x.query_id)
    n_scoreddocs = len(scoreddoc_bm25_sorted)
    n_batches = math.ceil(n_scoreddocs / batch_size)

    logging.log(logging.INFO, f"Computing first slice outputs for {n_scoreddocs} scoreddocs")

    docstore = dataset.docs_store()
    query_store = {query.query_id: query.text for query in dataset.queries_iter() if query.query_id in queries_subset}

    sim_scoreddocs = {}
    final_scoreddocs = []

    for sim_type in FIXED_SIM_TYPES:
        sim_scoreddocs[sim_type] = {i: [] for i in range(exit_level + 1)}

    for batch_n, batch in tqdm(enumerate(batched_iterable(scoreddoc_bm25_sorted, batch_size)), total=n_batches):
        batch_queries = [query_store[query_id] for query_id, doc_id, score in batch]
        batch_docs = [docstore.get(doc_id).text for query_id, doc_id, score in batch]
        tokenized = tokenizer(batch_queries, batch_docs, return_tensors="pt", padding="max_length",
                              truncation=True)
        tokenized = tokenized.to(device)

        outputs_ee = first_slice_model(**tokenized, all_similarities=True)

        for i, sim_matrix_dict in enumerate(outputs_ee):
            for sim_type, scores in sim_matrix_dict["similarities"].items():
                for j, (query_id, doc_id, score) in enumerate(batch):
                    sim_scoreddocs[sim_type][i].append(ScoredDoc(query_id, doc_id, scores[j].item()))

        # Compute the final scores
        scores_full = last_slice_model(embeddings=outputs_ee[-1]["hidden_states"],
                                       attention_mask=outputs_ee[-1]["attention_mask"])

        for i, (query_id, doc_id, _) in enumerate(batch):
            final_scoreddocs.append(ScoredDoc(query_id, doc_id, scores_full[i].item()))

    return sim_scoreddocs, final_scoreddocs


def main():
    parser = argparse.ArgumentParser(prog="RQ1: compute sim functions as metric of ranking quality",
                                     description="Evaluate the ranking quality of sim_function of a model on a specfied dataset.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", type=str, default='veneres/monobert', help="The model name")
    parser.add_argument("--scoreddoc_file",
                        type=str,
                        default=None,
                        help="The scoreddoc file")
    parser.add_argument("--exit_level", type=str, default=13, help="The exit level")
    parser.add_argument("--batch_size", type=int, default=64, help="The batch size")
    parser.add_argument("--device", type=str, default="cuda", help="The device")
    parser.add_argument("--dataset", type=str, default="msmarco-passage/train/judged", help="The dataset")
    parser.add_argument("--output_dir",
                        type=str,
                        default="results",
                        help="The output directory, the reuslts will be saved in a csv file (scoreddocs_l{exit_level}.csv)")

    args = parser.parse_args()

    dataset_name = args.dataset

    model_name = args.model_name

    scoreddoc_file = args.scoreddoc_file

    dataset = ir_datasets.load(dataset_name)

    if scoreddoc_file is not None:
        scoreddoc = pd.read_csv(scoreddoc_file, dtype={"query_id": str, "doc_id": str, "score": float})
    else:  # get the ir dataset scoreddoc
        scoreddoc = pd.DataFrame(dataset.scoreddocs_iter())

    exit_level = args.exit_level
    batch_size = args.batch_size
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        scoreddocs, final_scoreddocs = compute_first_slice_outputs(model_name,
                                                                   dataset,
                                                                   scoreddoc,
                                                                   tokenizer,
                                                                   exit_level,
                                                                   batch_size,
                                                                   device)

    pd.DataFrame(final_scoreddocs).to_csv(output_dir / f"scoreddocs.csv", index=False)

    for sim_type, sim_type_scoreddocs in scoreddocs.items():
        (output_dir / sim_type).mkdir(parents=True, exist_ok=True)
        for level, scored_docs_level in sim_type_scoreddocs.items():
            pd.DataFrame(scored_docs_level).to_csv(output_dir / sim_type / f"scoreddocs_l{level}.csv", index=False)


if __name__ == '__main__':
    main()
