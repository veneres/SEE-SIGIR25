import argparse
import json

import ir_datasets
import pandas as pd
import torch
import os

import transformers
from ir_measures import ScoredDoc
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Avoid warning and possible deadlocks...
# "huggingface/tokenizers: The current process just got forked, after parallelism has already been used.
# Disabling parallelism to avoid deadlocks..."
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class IRDatasetScoreddoc(Dataset):
    def __init__(self,
                 ir_dataset_name: str,
                 tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase):
        self.dataset = ir_datasets.load(ir_dataset_name)
        self.scoreddocs = list(self.dataset.scoreddocs_iter())
        print("Creating doc store...")
        self.dataset_docstore = self.dataset.docs_store()

        print("Creating queries_store...")
        self.dataset_queriestore = {query.query_id: query.text for query in tqdm(self.dataset.queries_iter())}

        print("Docstore and queries_store created")

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.scoreddocs)

    def __getitem__(self, idx):
        query_id, doc_id, _ = self.scoreddocs[idx]  # query_id, doc_id, score

        query_text = self.dataset_queriestore[query_id]
        doc_text = self.dataset_docstore.get(doc_id).text

        tokenized_text = self.tokenizer(query_text,
                                        doc_text,
                                        return_tensors="pt",
                                        padding="max_length",
                                        truncation=True)

        tokenized_text = {k: v[0] for k, v in tokenized_text.items()}
        tensor_qid = torch.tensor(int(query_id))
        tensor_did = torch.tensor(int(doc_id))
        return {**tokenized_text, "query_id": tensor_qid, "doc_id": tensor_did}


class SpladeRUN(Dataset):
    def __init__(self,
                 ir_dataset_name: str,
                 tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
                 splade_run_path: str):
        with open(splade_run_path, "r") as f:
            splade_run_json = json.load(f)
            self.splade_run = []  # splade run will contain tuples of (query_id, doc_id, score)
            for query_id, docs in splade_run_json.items():
                for doc_id, score in docs.items():
                    self.splade_run.append((query_id, doc_id, score))
        self.dataset = ir_datasets.load(ir_dataset_name)
        print("Creating doc store...")
        self.dataset_docstore = self.dataset.docs_store()

        print("Creating queries_store...")
        self.dataset_queriestore = {query.query_id: query.text for query in tqdm(self.dataset.queries_iter())}

        print("Docstore and queries_store created")

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.splade_run)

    def __getitem__(self, idx):
        query_id, doc_id, score = self.splade_run[idx]

        query_text = self.dataset_queriestore[query_id]
        doc_text = self.dataset_docstore.get(doc_id).text

        tokenized_text = self.tokenizer(query_text,
                                        doc_text,
                                        return_tensors="pt",
                                        padding="max_length",
                                        truncation=True)

        tokenized_text = {k: v[0] for k, v in tokenized_text.items()}
        tensor_qid = torch.tensor(int(query_id))
        tensor_did = torch.tensor(int(doc_id))
        return {**tokenized_text, "query_id": tensor_qid, "doc_id": tensor_did}


class ASNQDev(Dataset):
    def __init__(self,
                 asnq_path: str,
                 tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase):
        self.dataset = pd.read_csv(asnq_path, sep="\t", header=None,
                                   names=["qid", "pid", "question", "answer", "label"])

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        query_text = self.dataset.iloc[idx]["question"]
        doc_text = self.dataset.iloc[idx]["answer"]
        query_id = self.dataset.iloc[idx]["qid"]
        doc_id = self.dataset.iloc[idx]["pid"]

        tokenized_text = self.tokenizer(query_text,
                                        doc_text,
                                        return_tensors="pt",
                                        padding="max_length",
                                        truncation=True)

        tokenized_text = {k: v[0] for k, v in tokenized_text.items()}
        tensor_qid = torch.tensor(int(query_id))
        tensor_did = torch.tensor(int(doc_id))
        return {**tokenized_text, "query_id": tensor_qid, "doc_id": tensor_did}


def main():
    parser = argparse.ArgumentParser(description=
                                     """
                                    Create csv with the score for each query document pair from a given dataset.
                                    """,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("model", type=str, help="argument of AutoModelForSequenceClassification.from_pretrained()")

    parser.add_argument("out", type=str, help="Out file path containing the resulting dataframe")

    parser.add_argument("--ir_dataset",
                        type=str,
                        help="Dataset to use from ir dataset, to be specified if msmarco is used",
                        default=None)

    parser.add_argument("--splade_run", type=str, help="Path to splade run file")
    parser.add_argument("--batch_size", type=int, help="Batch size to use, default 1024", default=1024)
    parser.add_argument("--device", type=str, help="Device to use", default="cuda")

    args = parser.parse_args()

    batch_size = args.batch_size

    output = args.out

    dataset_name = args.ir_dataset

    device = torch.device(args.device)

    model = torch.nn.DataParallel(AutoModelForSequenceClassification.from_pretrained(args.model))

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except OSError:
        print("WARNING: tokenizer not found, using default tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model.to(device)

    model.eval()

    # Collection of QueryDocRel used to create the final df
    entries_df = []

    if dataset_name is not None:
        if args.splade_run is not None:
            print(f"Using SpladeRUN from {args.splade_run}")
            pt_dataset = SpladeRUN(dataset_name, tokenizer, args.splade_run)
        else:
            pt_dataset = IRDatasetScoreddoc(dataset_name, tokenizer)
    else:
        pt_dataset = ASNQDev(args.dev_dataset, tokenizer)

    dev_dataloader = DataLoader(pt_dataset, batch_size=batch_size, shuffle=False)

    # list to store all the predictions
    with torch.no_grad():
        with tqdm(dev_dataloader, unit="batch", total=len(dev_dataloader)) as tqdm_wrapper_ds_loader:
            for batch in tqdm_wrapper_ds_loader:
                batch_forward = {k: v.to(device) for k, v in batch.items() if
                                 k not in ["query_id", "doc_id"]}

                batch_queries_ids = batch["query_id"]
                batch_docs_ids = batch["doc_id"]
                outputs = model(**batch_forward)
                if outputs.logits.shape[1] == 1:  # for naver/trecdl22-crossencoder-electra
                    scores = torch.sigmoid(outputs.logits).squeeze()
                else:  # for all the other models
                    scores = torch.softmax(outputs.logits, dim=1)[:, 1]

                for i, score in enumerate(scores):
                    score = score.item()
                    query_id = batch_queries_ids[i].item()
                    doc_id = batch_docs_ids[i].item()

                    entries_df.append(ScoredDoc(query_id=query_id, doc_id=doc_id, score=score))

    pd.DataFrame(entries_df).to_csv(output, index=False)


if __name__ == '__main__':
    main()
