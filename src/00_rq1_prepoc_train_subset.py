from pathlib import Path

import ir_datasets
import numpy as np
from tqdm import tqdm
from ir_measures import ScoredDoc
import pandas as pd
import argparse
import logging

SAMPLE_SIZE = 1000


def main():
    parser = argparse.ArgumentParser(description="Compute a subsample of the train for initial exploration of the "
                                                 "technique.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--output_file", help="File where to store the run", type=str,
                        default="/data/rq1/sample_scoreddocs_train.csv")
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)


    args = parser.parse_args()

    output_file = args.output_file
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    dataset = ir_datasets.load("msmarco-passage/train/judged")
    query_ids = [query.query_id for query in dataset.queries_iter()]

    # Take a random sample of SAMPLE_SIZE queries
    np.random.seed(42)
    sample_query_ids = np.random.choice(query_ids, SAMPLE_SIZE, replace=False).tolist()

    logging.info(f"Correctly sampled {len(sample_query_ids)} query ids")

    scoreddoc_bm25 = []

    for elem in tqdm(dataset.scoreddocs_iter(), total=dataset.scoreddocs_count()):
        if elem.query_id in sample_query_ids:
            scoreddoc_bm25.append(ScoredDoc(query_id=elem.query_id, doc_id=elem.doc_id, score=0))

    pd.DataFrame(scoreddoc_bm25).to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
