from collections import namedtuple
from pathlib import Path

import numpy as np
import argparse
import pandas as pd
import ir_datasets
import ir_measures
from ir_measures import Qrel
from tqdm import tqdm

SCORED_DATASET_FILENAME = 'scoreddocs.csv'

SCORED_DATASET_PREFIX = "scoreddocs_l"

DATASET = "msmarco-passage/train/judged"

ResRow = namedtuple('ResRow', ["exit_level", "ndcg", "delta_ndcg", "recall"])


def evaluate_run(data_path: Path,
                 recall_dataset: pd.DataFrame,
                 model_scored_dataset: pd.DataFrame,
                 judgements: pd.DataFrame,
                 nk: int,
                 rk: int,
                 n_levels: int) -> pd.DataFrame:
    ndcg = ir_measures.nDCG @ nk
    recall = ir_measures.Recall @ rk
    base_ndcg = ndcg.calc_aggregate(qrels=judgements, run=model_scored_dataset)
    base_recall = recall.calc_aggregate(qrels=recall_dataset, run=model_scored_dataset)
    res = []
    res.append(ResRow("full", base_ndcg, 0, base_recall))

    for depth in tqdm(range(n_levels), desc='Computing results for different depths'):
        run_filename = f'{SCORED_DATASET_PREFIX}{depth}.csv'
        scored_dataset = pd.read_csv(data_path / run_filename)
        scored_dataset['query_id'] = scored_dataset['query_id'].astype(str)
        scored_dataset['doc_id'] = scored_dataset['doc_id'].astype(str)
        measured_ndcg = ndcg.calc_aggregate(qrels=judgements, run=scored_dataset)
        measured_recall = recall.calc_aggregate(qrels=recall_dataset, run=scored_dataset)
        delta_ndcg = (base_ndcg - measured_ndcg) / base_ndcg
        res.append(ResRow(depth, measured_ndcg, delta_ndcg, measured_recall))

    res = pd.DataFrame(res)
    return res


def main():
    parser = argparse.ArgumentParser(prog="RQ1: Ranking quality evaluation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path",
                        type=str,
                        default='/see-data/rq1/msmarco-train-sample/veneres/monobert/amax',
                        help="The data path ot ")
    parser.add_argument("--nk", type=int, default=10, help="The k value for ndcg@k")
    parser.add_argument("--rk", type=int, default=100, help="The k value for recall@k")
    parser.add_argument("--output", type=str, default="stats.csv", help="Output file name")

    args = parser.parse_args()
    data_path = Path(args.data_path)
    nk = args.nk
    rk = args.rk
    output = args.output

    reference_dataset = ir_datasets.load(DATASET)

    full_scoreddoc = pd.read_csv(data_path.parent / SCORED_DATASET_FILENAME, dtype={'query_id': str, 'doc_id': str})

    queries_subset = set(full_scoreddoc['query_id'].unique())

    qrels_judged = pd.DataFrame([qrel for qrel in reference_dataset.qrels_iter() if qrel.query_id in queries_subset])

    recall_scoreddocs = []
    for query_id in tqdm(full_scoreddoc['query_id'].unique(), desc='Computing recall dataset'):
        full_scored_query = full_scoreddoc.query(f'query_id == @query_id')
        query_top = np.zeros(full_scored_query['score'].values.size, dtype=int)
        actual_top_k = np.argsort(full_scored_query['score'].values)[-10:]
        query_top[actual_top_k] = 1
        for i, doc_id in enumerate(full_scored_query['doc_id']):
            sd = Qrel(query_id, doc_id, query_top[i])
            recall_scoreddocs.append(sd)

    recall_based_dataset = pd.DataFrame(recall_scoreddocs)

    res = evaluate_run(data_path=data_path,
                       recall_dataset=recall_based_dataset,
                       model_scored_dataset=full_scoreddoc,
                       judgements=qrels_judged,
                       nk=nk,
                       rk=rk,
                       n_levels=13)

    res.to_csv(data_path / output, index=False)


if __name__ == '__main__':
    main()
