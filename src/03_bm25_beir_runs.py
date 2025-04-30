
import argparse
from pathlib import Path

import ir_datasets
import ir_measures
import pandas as pd

import Stemmer
from ir_measures import ScoredDoc, RR

import bm25s
DATASETS = [
    "beir/arguana",
    "beir/climate-fever",
    "beir/cqadupstack/android",
    "beir/cqadupstack/english",
    "beir/cqadupstack/gaming",
    "beir/cqadupstack/gis",
    "beir/cqadupstack/mathematica",
    "beir/cqadupstack/physics",
    "beir/cqadupstack/programmers",
    "beir/cqadupstack/stats",
    "beir/cqadupstack/tex",
    "beir/cqadupstack/unix",
    "beir/cqadupstack/webmasters",
    "beir/cqadupstack/wordpress",
    "beir/dbpedia-entity/test",
    "beir/fever/test",
    "beir/fiqa/test",
    "beir/hotpotqa/test",
    "beir/msmarco/test",
    "beir/nfcorpus/test",
    "beir/nq",
    "beir/quora/test",
    "beir/scidocs",
    "beir/scifact/test",
    "beir/trec-covid",
    "beir/webis-touche2020/v2"

]


def postprocess_results_for_eval(results, scores, query_ids):
    """
    Given the queried results and scores output by BM25S, postprocess them
    to be compatible with ir_measures evaluation functions.
    query_ids is a list of query ids in the same order as the results.
    """

    run = []
    for i, qid in enumerate(query_ids):
        for j in range(len(results[i])):
            run.append(ScoredDoc(query_id=qid, doc_id=results[i][j], score=scores[i][j]))

    return run


def run_benchmark(dataset_name: str, save_dir: Path):
    #### Download dataset and unzip the dataset
    dataset = ir_datasets.load(dataset_name)

    corpus_ids, corpus_lst = [], []
    for doc in dataset.docs_iter():
        title = doc.title if hasattr(doc, 'title') else ""
        body = doc.text
        corpus_lst.append(f"{title} {body}".strip())
        corpus_ids.append(doc.doc_id)

    qids, queries_lst = [], []
    for query in dataset.queries_iter():
        query_text = query.text.strip()
        queries_lst.append(query_text)
        qids.append(query.query_id)

    stemmer = Stemmer.Stemmer("english")

    corpus_tokens = bm25s.tokenize(
        corpus_lst, stemmer=stemmer, leave=False
    )

    query_tokens = bm25s.tokenize(
        queries_lst, stemmer=stemmer, leave=False
    )

    model = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
    model.index(corpus_tokens, leave_progress=False)

    ############## BENCHMARKING BEIR HERE ##############
    queried_results, queried_scores = model.retrieve(
        query_tokens, corpus=corpus_ids, k=1000, n_threads=4
    )

    scoreddoc = postprocess_results_for_eval(queried_results, queried_scores, qids)

    eval_results = ir_measures.calc_aggregate([RR @ 10],
                                              dataset.qrels_iter(),
                                              scoreddoc)
    print(f"Results for {dataset_name}: {eval_results}")

    output_file = save_dir / f"{dataset_name.replace('/', '_')}.csv"
    print("Saving to: ")
    print(output_file.resolve())
    pd.DataFrame(scoreddoc).to_csv(save_dir / f"{dataset_name.replace('/', '_')}.csv", index=False)

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="/data/beir-bm25-runs")
    args = parser.parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        run_benchmark(dataset, save_dir)


if __name__ == "__main__":
    main()
