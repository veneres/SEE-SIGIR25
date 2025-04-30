import contextlib
from pathlib import Path

import ir_datasets
from ir_datasets.formats import TsvDocs, TsvQueries, TrecQrels, TrecScoredDocs
from ir_datasets.util import TarExtract


# Modified from https://github.com/seanmacavaney/dummy-irds-ext/blob/master/dummy_irds_ext/__init__.py


class PreprocQrel:
    def __init__(self, dev_streamer, path_in: Path, out_path: Path):
        self._path_in = path_in
        self._out_path = out_path
        self._dev_streamer = dev_streamer

    @contextlib.contextmanager
    def stream(self):
        # TODO change and use the streamer in the loop
        self._dev_streamer.verify()

        with open(self._out_path / "qrels.tsv", 'w') as fout:
            with open(self._path_in / "dev_preproc.tsv") as fin:
                for line in fin:
                    qid, pid, label = line.strip().split('\t')
                    if label == "1":
                        fout.write(f"{qid}\t{pid}\t1\n")

        yield open(self._out_path / "qrels.tsv", "rb")


class PreprocDev:
    def __init__(self, dev_streamer, path_in: Path, out_path: Path):
        self._path_in = path_in
        self._out_path = out_path
        self._dev_streamer = dev_streamer

    @contextlib.contextmanager
    def stream(self):
        # TODO change and use the streamer in the loop
        self._dev_streamer.verify()
        query_count, doc_count = 0, 0
        query2id = {}
        doc2id = {}
        with open(self._path_in / "dev.tsv", 'r') as fin:
            with open(self._out_path / "dev_preproc.tsv", 'w') as dev_out_file:
                with open(self._out_path / "dev_preproc_queries.tsv", 'w') as queries_out_file:
                    with open(self._out_path / "dev_preproc_docs.tsv", 'w') as docs_out_file:
                        for line in fin:
                            query, doc, label = line.strip().split('\t')
                            if query in query2id:
                                qid = query2id[query]
                            else:
                                qid = query_count
                                query2id[query] = qid
                                query_count += 1
                                queries_out_file.write(f"{qid}\t{query}\n")
                            if doc in doc2id:
                                pid = doc2id[doc]
                            else:
                                pid = doc_count
                                doc2id[doc] = pid
                                doc_count += 1
                                docs_out_file.write(f"{pid}\t{doc}\n")
                            if label == '4':
                                label = 1
                            else:
                                label = 0
                            dev_out_file.write(f"{qid}\t{pid}\t{label}\n")
        yield open(self._out_path / "dev_preproc.tsv", 'rb')


class PreprocDocs:
    def __init__(self, dev_streamer, path_in: Path, out_path: Path):
        self._path_in = path_in
        self._out_path = out_path
        self._dev_streamer = dev_streamer

    @contextlib.contextmanager
    def stream(self):
        self._dev_streamer.verify()
        yield open(self._out_path / "dev_preproc_docs.tsv", 'rb')


class PreprocQueries:
    def __init__(self, dev_streamer, path_in: Path, out_path: Path):
        self._path_in = path_in
        self._out_path = out_path
        self._dev_streamer = dev_streamer

    @contextlib.contextmanager
    def stream(self):
        self._dev_streamer.verify()
        yield open(self._out_path / "dev_preproc_queries.tsv", 'rb')


NAME = 'asnq'

# What to the relevance levels in qrels mean?
QREL_DEFS = {
    1: 'relevant',
    0: 'not relevant',
}

# Specify where to find the content. Here it's just from the repository, but it could be anywhere.
DL = ir_datasets.util.RequestsDownload("https://d3t7erp6ge410c.cloudfront.net/tanda-aaai-2020/data/asnq.tar")

# where the content is cached
base_path = ir_datasets.util.home_path() / NAME

raw_data = ir_datasets.util.Cache(DL, base_path / 'asnq.tar')

# Request immediate download of the content to unpack it

unpacked_train = ir_datasets.util.Cache(TarExtract(raw_data, "data/asnq/train.tsv", compression=""),
                                        base_path / "train.tsv")

unpacked_dev = ir_datasets.util.Cache(TarExtract(raw_data, "data/asnq/dev.tsv", compression=""),
                                      base_path / "dev.tsv")

# preprocessed_train = ir_datasets.util.Cache(TarExtract(raw_data, "data/asnq/train.tsv", compression='x'),
#                                       base_path / "train_.tsv")

# unpacked_train.verify()


preproc_dev = ir_datasets.util.Cache(PreprocDev(unpacked_dev, base_path, base_path), base_path / "dev_preproc.tsv")

preproc_qrel = ir_datasets.util.Cache(PreprocQrel(preproc_dev, base_path, base_path), base_path / "qrels.tsv")


# Dataset definition: it provides docs, queries, and qrels
dataset = ir_datasets.Dataset(
    TsvDocs(ir_datasets.util.Cache(PreprocDocs(preproc_dev, base_path, base_path), base_path / "dev_preproc_docs.tsv")),
    TsvQueries(ir_datasets.util.Cache(PreprocQueries(preproc_dev, base_path, base_path), base_path / 'dev_preproc_queries.tsv')),
    TrecScoredDocs(preproc_dev),
    TrecQrels(preproc_qrel, QREL_DEFS, format_3col=True),
)

# Register the dataset with ir_datasets
ir_datasets.registry.register(NAME, dataset)
