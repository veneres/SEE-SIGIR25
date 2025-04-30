# modified from https://github.com/castorini/earlyexiting-monobert/blob/main/data/asnq/preprocess.py

from collections import defaultdict
import argparse
from pathlib import Path
from tqdm import tqdm


def pick_train(path_to_asnq: Path, out_path: Path):
    # each query has no more than 1 pos and neg documents
    collection = defaultdict(list)
    collected = defaultdict(
        list)  # key: query; value: a list of 0 and 1 (whether negative and positive documents collected
    with open(path_to_asnq / "train.tsv") as fin:
        for line in tqdm(fin):
            query, doc, label = line.strip().split('\t')
            if label == '4' and 1 not in collected[query]:
                collection[query].append([doc, 1])
                collected[query].append(1)
            elif label in ['0', '1', '2'] and 0 not in collected[query]:
                collection[query].append([doc, 0])
                collected[query].append(0)

    with open(out_path / "train.tsv", 'w') as fout:
        print('Question\tAnswer\tLabel', file=fout)
        for query, qlist in collection.items():
            for (doc, label) in qlist:
                print(f'{query}\t{doc}\t{label}', file=fout)


def process_dev(path_to_asnq: Path, out_path: Path):
    query_count, doc_count = 0, 0
    query2id = {}
    doc2id = {}
    with open(path_to_asnq / "dev.tsv") as fin:
        with open(out_path / "dev.tsv", 'w') as fout:
            for line in fin:
                query, doc, label = line.strip().split('\t')
                if query in query2id:
                    qid = query2id[query]
                else:
                    qid = query_count
                    query2id[query] = qid
                    query_count += 1
                if doc in doc2id:
                    pid = doc2id[doc]
                else:
                    pid = doc_count
                    doc2id[doc] = pid
                    doc_count += 1
                if label == '4':
                    label = 1
                else:
                    label = 0
                print('{}\t{}\t{}\t{}\t{}'.format(
                    qid,
                    pid,
                    query,
                    doc,
                    label), file=fout)


def get_qrel(out_path: Path):
    with open(out_path / "asnq.qrel.dev.tsv", 'w') as fout:
        with open(out_path / "dev.tsv") as fin:
            for line in fin:
                qid, pid, query, doc, label = line.strip().split('\t')
                if label == '1':
                    print('{}\t0\t{}\t1'.format(qid, pid), file=fout)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', type=str, help='path to the original data')
    parser.add_argument('output_path', type=str, help='path to the output data')

    args = parser.parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    pick_train(input_path, output_path)
    process_dev(input_path, output_path)
    get_qrel(output_path)

if __name__ == '__main__':
    main()
