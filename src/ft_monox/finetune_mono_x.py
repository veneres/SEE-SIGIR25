import argparse
from collections import namedtuple, defaultdict

import numpy as np
import pandas as pd
import torch

from pathlib import Path

import transformers
# from datasets import load_dataset
from torch import nn
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
from torch.utils.data import Dataset
import ir_datasets

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

MODEL_MAPPING = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "electra": "google/electra-base-discriminator"
}


class IRDatasetMonoLoader(Dataset):
    def __init__(self,
                 ir_dataset_name: str,
                 tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
                 irdataset_pt_file_name: str):
        self.triples_pt = torch.load(irdataset_pt_file_name)  # qrels preprocessed by ir_datasets for re-ranking
        self.dataset = ir_datasets.load(ir_dataset_name)
        print("Creating doc store...")
        self.dataset_docstore = self.dataset.docs_store()

        print("Creating queries_store...")
        self.dataset_queriestore = {query.query_id: query.text for query in tqdm(self.dataset.queries_iter())}

        print("Docstore and queries_store created")

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.triples_pt)

    def get_labels(self):
        return self.triples_pt[:, 2]

    def __getitem__(self, idx):
        qrel = self.triples_pt[idx, :]

        query_id = str(qrel[0].item())  # First element is query_id
        doc_id = str(qrel[1].item())  # Second element is doc_id
        relevance = qrel[2].item()  # Third element is relevance

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
        return {**tokenized_text, "labels": relevance, "query_id": tensor_qid, "doc_id": tensor_did}


class IRDatasetMonoLoaderUnique(Dataset):
    def __init__(self,
                 ir_dataset_name: str,
                 tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase):

        self.dataset = ir_datasets.load(ir_dataset_name)

        self.triples = []

        used_query_rel_pair = set()

        for query_id, rel_doc_id, neg_doc_id in tqdm(self.dataset.docpairs_iter(), total=self.dataset.docpairs_count()):
            if (query_id, rel_doc_id) in used_query_rel_pair:
                continue
            self.triples.append((query_id, rel_doc_id, 1))
            self.triples.append((query_id, neg_doc_id, 0))
            used_query_rel_pair.add((query_id, rel_doc_id))

        print("Creating doc store...")
        self.dataset_docstore = self.dataset.docs_store()

        print("Creating queries_store...")
        self.dataset_queriestore = {query.query_id: query.text for query in tqdm(self.dataset.queries_iter())}

        print("Docstore and queries_store created")

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.triples)

    def get_labels(self):
        return self.triples[:, 2]

    def __getitem__(self, idx):
        query_id, doc_id, relevance = self.triples[idx]

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
        return {**tokenized_text, "labels": relevance, "query_id": tensor_qid, "doc_id": tensor_did}


class SpladeDatasetLoader(Dataset):
    TripleEntry = namedtuple("TripleEntry", ["query_id", "doc_id", "relevance"])

    def __init__(self,
                 ir_dataset_name: str,
                 triples_run_file: str,
                 tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase):
        print("Loading splade triples...")
        self.triples = pd.read_csv(triples_run_file, dtype={"query_id": str, "doc_id": str, "relevance": int})

        self.dataset = ir_datasets.load(ir_dataset_name)

        print("Creating doc store...")
        self.dataset_docstore = self.dataset.docs_store()

        print("Creating queries_store...")
        self.dataset_queriestore = {query.query_id: query.text for query in tqdm(self.dataset.queries_iter())}

        print("Docstore and queries_store created")

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.triples)

    def get_labels(self):
        return self.triples[:, 2]

    def __getitem__(self, idx):
        qrel = self.triples.iloc[idx]

        query_id = qrel.query_id
        doc_id = qrel.doc_id
        relevance = qrel.relevance

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
        return {**tokenized_text, "labels": relevance, "query_id": tensor_qid, "doc_id": tensor_did}


class ASNQCustomDataloader(Dataset):
    def __init__(self,
                 asnq_path: str,
                 tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase):
        with open(file=asnq_path, mode="r") as f:
            self.dataset = f.readlines()
            # question \t answer \t label
        self.dataset = [line.strip().split("\t") for line in self.dataset]
        self.dataset = self.dataset[1:]  # remove header
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        query_text = self.dataset[idx][0]
        doc_text = self.dataset[idx][1]
        relevance = int(self.dataset[idx][2])

        tokenized_text = self.tokenizer(query_text,
                                        doc_text,
                                        return_tensors="pt",
                                        padding="max_length",
                                        truncation=True)

        tokenized_text = {k: v[0] for k, v in tokenized_text.items()}
        return {**tokenized_text, "labels": relevance}


class TrainerMonoX:

    def __init__(self,
                 model: torch.nn.Module,
                 train_dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 device_id: int,
                 lr_scheduler: torch.optim.lr_scheduler,
                 save_every: int,
                 output_dir: Path):
        self.model = model.to(device_id)
        self.model = DDP(model, device_ids=[device_id])
        self.device = torch.device(device_id)
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.save_every = save_every
        self.lr_scheduler = lr_scheduler
        self.window_loss = []
        self.save_every = save_every
        self.output_dir = output_dir

    def _run_batch(self, batch_forward, labels):
        self.optimizer.zero_grad()

        outputs = self.model(**batch_forward)

        logits = outputs.logits.to(self.device)
        cel_w = nn.CrossEntropyLoss().to(self.device)
        loss_w = cel_w(logits, labels)
        loss_w.backward()

        self.optimizer.step()
        self.lr_scheduler.step()

        self.window_loss.append(loss_w.item())

    def _run_epoch(self, epoch):
        step_n = 0
        with tqdm(self.train_dataloader, unit="batch", total=len(self.train_dataloader)) as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                batch_forward = {k: v.to(self.device) for k, v in batch.items() if
                                 k not in ["query_id", "doc_id", "labels"]}
                labels = batch["labels"].to(self.device)
                self._run_batch(batch_forward, labels)

                if len(self.window_loss) < 100:  # window loss updated by run_batch
                    tepoch.set_postfix(loss="---")
                else:
                    tepoch.set_postfix(loss=np.mean(self.window_loss))
                    self.window_loss = self.window_loss[1:]

                step_n += 1
                if self.device.index == 0 and step_n % self.save_every == 0:
                    self.save_checkpoint(epoch, step_n, self.optimizer, self.lr_scheduler)
        if self.device.index == 0:
            self.save_checkpoint(epoch, step_n, self.optimizer, self.lr_scheduler)

    def save_checkpoint(self, epoch: int, step_n: int, optimizer, scheduler):
        filename = f"{str(epoch)}_{str(step_n)}"
        output_dir_step = self.output_dir / filename
        print(f"Saving checkpoint in: {output_dir_step}")
        self.model.module.save_pretrained(output_dir_step)
        torch.save(optimizer.state_dict(), output_dir_step / "optimizer.pt")
        torch.save(scheduler.state_dict(), output_dir_step / "scheduler.pt")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)


def ddp_setup(rank: int, world_size: int):
    """
    Args:
           rank: Unique identifier of each process
          world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size, group_name="veneres")
    torch.cuda.set_device(rank)


def main(rank: int,
         world_size: int,
         llm: str,
         train_dataset: str,
         train_pp_ds: str,
         batch_size: int,
         output_dir: str,
         save_every: int,
         num_epochs: int,
         asnq_path: str,
         splade_triples_path: str):
    ddp_setup(rank=rank, world_size=world_size)

    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if llm not in MODEL_MAPPING and not args.continue_training:
        raise Exception(f"LLM passed via --lm argument non under consideration. Value passed: {llm}")

    model_path = MODEL_MAPPING[llm]  # hugging face large language model name

    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Load tokenizer and datasets

    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAPPING[llm])

    if train_dataset == "asnq":
        train_dataset = ASNQCustomDataloader(asnq_path, tokenizer)
    elif splade_triples_path is not None:
        train_dataset = SpladeDatasetLoader(train_dataset, splade_triples_path, tokenizer)
    else:
        train_dataset = IRDatasetMonoLoader(train_dataset, tokenizer, irdataset_pt_file_name=train_pp_ds)

    seed = 2147483647

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  sampler=DistributedSampler(train_dataset, seed=seed, num_replicas=world_size,
                                                             rank=rank, shuffle=True, drop_last=False))

    optimizer = AdamW(model.parameters(), lr=1e-6)

    num_warmup_steps = 1e4

    # Print arguments

    if rank == 0:
        print(f"{batch_size=}")

        print(f"{num_epochs=}")

        print(f"{train_dataset=}")

        print(f"{train_pp_ds=}")

        print(f"{llm}")

        print(f"{output_dir}")

        print(f"{num_warmup_steps=}")

    else:
        print(f"Device with rank {rank} running")

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=len(train_dataloader)
    )
    model.train()

    trainer = TrainerMonoX(model=model,
                           train_dataloader=train_dataloader,
                           optimizer=optimizer,
                           device_id=rank,
                           lr_scheduler=lr_scheduler,
                           save_every=save_every,
                           output_dir=output_dir)

    trainer.train(max_epochs=num_epochs)
    destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine tune monox model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("llm", type=str, help="""
                                                    Large Language Model (LLM) to be used.
                                                    Possible choice:
                                                    - bert
                                                    - roberta
                                                    - electra
                                                    """)
    parser.add_argument("train_dataset",
                        type=str,
                        help="Train dataset to use from ir dataset")

    parser.add_argument("--train_pp_ds", type=str, help="Path to train pre-processed dataset, needed for ir datasets",
                        default=None)
    parser.add_argument("--batch_size", type=int, help="Batch size to use", default=128)

    parser.add_argument("--output_dir", type=str, help="Default as the same string as llm argument", default=None)
    parser.add_argument("--save_every", type=int, help="Number of training steps to wait before saving the checkpoint",
                        default=100000)
    parser.add_argument("--num_epochs", type=int, help="Number of epochs", default=5)
    parser.add_argument("--asnq_path", type=str, help="Path to asnq dataset", default=None)
    parser.add_argument("--splade_triples_path", type=str, help="Path to splade triples", default=None)

    args = parser.parse_args()

    llm = args.llm
    train_dataset = args.train_dataset
    train_pp_ds = args.train_pp_ds
    batch_size = args.batch_size
    output_dir = args.output_dir
    save_every = args.save_every
    num_epochs = args.num_epochs
    asnq_path = args.asnq_path
    splade_triples_path = args.splade_triples_path

    world_size = torch.cuda.device_count()

    mp.spawn(main,
             args=(world_size,
                   llm,
                   train_dataset,
                   train_pp_ds,
                   batch_size,
                   output_dir,
                   save_every,
                   num_epochs,
                   asnq_path,
                   splade_triples_path),
             nprocs=world_size)
