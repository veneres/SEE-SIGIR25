# Here we simulate the experiments for RQ2, without  running the real model, we use the precomputed maxsims and scores
# the speedup is based on the percentage of pairs that are not processed by the second part of the model
# with thw assumption that traversing the model is a linear cost proportional to the number of pairs and layers.
from pathlib import Path

import numpy as np
import pandas as pd
import ir_datasets

import asnq_dataset  # do not delete even if it is not used

import ir_measures
from tqdm import tqdm
from utils import EarlyExitMode, apply_EE
from joblib import Parallel, delayed
import argparse




MODEL_KEYNAMES = ["veneres/monobert"]
DATASET_RQ1 = "msmarco-train-sample"

# Fixed Exit modes
EXIT_MODES = [EarlyExitMode.STATIC_THRESHOLD, EarlyExitMode.PROXIMITY_STATIC]

# Fixed parameters for the early exit
PARAMS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Fixed exit positions
EXIT_POSITIONS = [0, 1, 2]

# Fixed cutoff
MK = 10

MODEL_NUMBER_OF_LAYERS = {
    "veneres/monobert": 12
}


def dataset_mask(scored_dataset, reference_dataset, exit_mode, param, ek):
    masked_df = reference_dataset.copy()
    # Uniques are returned in order of appearance. Hash table-based unique, therefore does NOT sort.
    # https: // pandas.pydata.org / pandas - docs / stable / reference / api / pandas.Series.unique.html
    query_ids = masked_df['query_id'].unique()

    # list of tuples (query_id, [scores])
    queries_scores = []

    for query_id in query_ids:
        query_df = scored_dataset[scored_dataset['query_id'] == query_id]
        reference_query = reference_dataset[reference_dataset['query_id'] == query_id]
        model_scores = reference_query['score'].values
        maxsim_scores = query_df['score_scale'].values
        mask = apply_EE(maxsim_scores, exit_mode, param, k=ek)
        query_scores = np.zeros(len(model_scores))
        query_scores[mask] = model_scores[mask]
        query_scores[~mask] = maxsim_scores[~mask] - 1000
        queries_scores += query_scores.tolist()

    masked_df['score'] = queries_scores

    return masked_df


def evaluate_early_exit(masked_dataset, reference, qrels, exit_layer, nlevels):
    ndgc_evaluator = ir_measures.nDCG @ MK
    mrr_evaluator = ir_measures.MRR @ MK
    ndcg_measures = [(elem.query_id, elem.value) for elem in ndgc_evaluator.iter_calc(qrels, masked_dataset)]
    mrr_measures = [(elem.query_id, elem.value) for elem in mrr_evaluator.iter_calc(qrels, masked_dataset)]
    data = []
    passed_docs = masked_dataset[masked_dataset['score'] > -900]
    for i, (query_id, query_ndcg) in enumerate(ndcg_measures):
        # query_id = measure.query_id
        npassed = len(passed_docs[passed_docs['query_id'] == query_id])

        reference_query = reference[reference['query_id'] == query_id]

        ref_ndcg = reference_query['ndcg'].values[0]
        ref_mrr = reference_query['mrr'].values[0]

        exit_ndcg = query_ndcg
        exit_mrr = mrr_measures[i][1]

        ndocs = len(reference_query)

        percentage_passed = npassed / ndocs

        cost = exit_layer * ndocs + (nlevels - exit_layer) * npassed  # removed cost of tokenizer

        base_cost = nlevels * ndocs
        if cost == 0:
            speedup = +np.inf
        else:
            speedup = base_cost / cost

        query_ev = [
            query_id,
            ref_ndcg,
            exit_ndcg,
            ref_mrr,
            exit_mrr,
            npassed,
            ndocs,
            percentage_passed,
            cost,
            base_cost,
            speedup,
        ]
        data.append(query_ev)
    columns = ['query_id', 'base_ndcg', 'exit_ndcg', 'base_mrr', 'exit_mrr', 'n_passed', 'n_total', 'percentage_passed',
               'cost', 'base_cost', 'speedup']
    return pd.DataFrame(data, columns=columns)


def compute_df(dataset_name: str,
               model_keyname: str,
               exit_mode: EarlyExitMode,
               exit_param: float,
               exit_position: int,
               nlevels: int,
               qrel_judged: pd.DataFrame,
               full_scored_dataset: pd.DataFrame,
               scored_dataset_layer: pd.DataFrame):
    ndcg_evaluator = ir_measures.nDCG @ MK
    mrr_evaluator = ir_measures.MRR()

    ndgcs = {elem.query_id: elem.value for elem in ndcg_evaluator.iter_calc(qrel_judged, full_scored_dataset)}
    mrrs = {elem.query_id: elem.value for elem in mrr_evaluator.iter_calc(qrel_judged, full_scored_dataset)}
    ndcg_to_append = []
    mrr_to_append = []
    for query_id in full_scored_dataset["query_id"]:
        ndcg_to_append.append(ndgcs[query_id])
        mrr_to_append.append(mrrs[query_id])

    full_scored_dataset["ndcg"] = ndcg_to_append
    full_scored_dataset["mrr"] = mrr_to_append

    masked_dataset = dataset_mask(scored_dataset_layer, full_scored_dataset, exit_mode, exit_param, ek=MK)
    ev_df = evaluate_early_exit(masked_dataset, full_scored_dataset, qrel_judged, exit_position, nlevels).reset_index(
        drop=True)
    ev_df['exit_position'] = exit_position
    ev_df['exit_k'] = MK
    ev_df['param'] = exit_param
    ev_df['mode'] = exit_mode.name
    ev_df['model'] = model_keyname
    ev_df['dataset'] = dataset_name
    return ev_df


def main():
    parser = argparse.ArgumentParser(prog="Ranking quality evaluation",
                                     description="Evaluate the ranking quality of a model on a dataset given a specific SEE"
                                                 "setting.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output_dir",
                        type=str,
                        default="/data/rq2",
                        help="The output folder for the results")
    parser.add_argument("--sim_type",
                        type=str,
                        default="amax",
                        help="The similarity type to use")
    parser.add_argument("--data_dir",
                        type=str,
                        default="/data/rq1",
                        help="The data directory containing the raw data")

    parser.add_argument("--n_jobs",
                        type=int,
                        default=64,
                        help="The number of jobs to run in parallel")

    parser.add_argument("--dataset",
                        type=str,
                        default="msmarco-passage/train/judged",
                        help="The dataset to use")

    args = parser.parse_args()

    output_dir = args.output_dir
    sim_type = args.sim_type
    data_dir = Path(args.data_dir)
    n_jobs = args.n_jobs
    dataset_name = args.dataset

    path_out = Path(output_dir)
    path_out.mkdir(parents=True, exist_ok=True)

    # Load all datasets and qrels in memory to avoid race conditions with the multiple process
    full_scored_datasets = {}
    scored_datasets_layer = {}
    qrels = {}
    for model_keyname in MODEL_KEYNAMES:
        for exit_position in EXIT_POSITIONS:
            # Load datasets
            reference_dataset = ir_datasets.load(dataset_name)

            path_prefix = data_dir / DATASET_RQ1 / model_keyname

            # Load scored dataset at layer depth
            run_filename = path_prefix / sim_type / f'scoreddocs_l{exit_position}.csv'
            scored_dataset_layer = pd.read_csv(run_filename, dtype={'query_id': str, 'doc_id': str})

            g = scored_dataset_layer.groupby('query_id')['score']
            min_, max_ = g.transform('min'), g.transform('max')
            scored_dataset_layer['score' + '_scale'] = (scored_dataset_layer['score'] - min_) / (max_ - min_)

            scored_dataset_filename = path_prefix / f"scoreddocs.csv"

            full_scored_dataset = pd.read_csv(scored_dataset_filename, dtype={'query_id': str, 'doc_id': str})

            key = f"{model_keyname}-{exit_position}"

            full_scored_datasets[key] = full_scored_dataset
            subset_query_ids = set(full_scored_dataset['query_id'].unique())
            qrels_subset = [qrel for qrel in reference_dataset.qrels_iter() if qrel.query_id in subset_query_ids]
            qrels[key] = pd.DataFrame(qrels_subset)
            scored_datasets_layer[key] = scored_dataset_layer

    # analysis of early exits (one per thread)
    evaluation_df = pd.DataFrame()
    list_of_params = []

    for model_keyname in MODEL_KEYNAMES:
        for exit_mode in EXIT_MODES:
            for exit_position in EXIT_POSITIONS:
                for param in PARAMS:
                    key = f"{model_keyname}-{exit_position}"
                    list_of_params.append({"dataset_name": dataset_name,
                                           "model_keyname": model_keyname,
                                           "exit_mode": exit_mode,
                                           "exit_param": param,
                                           "exit_position": exit_position,
                                           "nlevels": MODEL_NUMBER_OF_LAYERS[model_keyname],
                                           "full_scored_dataset": full_scored_datasets[key],
                                           "scored_dataset_layer": scored_datasets_layer[key],
                                           "qrel_judged": qrels[key]
                                           })

    ev_dfs = Parallel(n_jobs=n_jobs)(delayed(compute_df)(**params) for params in tqdm(list_of_params))
    # ev_dfs = [compute_df(**params) for params in tqdm(list_of_params)]
    for ev_df in ev_dfs:
        evaluation_df = pd.concat([evaluation_df, ev_df])
    print(evaluation_df.head())
    print(evaluation_df.describe())
    evaluation_df.to_csv(path_out / f'stats.csv', index=False)


if __name__ == '__main__':
    main()