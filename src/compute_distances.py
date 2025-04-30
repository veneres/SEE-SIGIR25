import argparse
import math
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import ir_datasets
from tqdm import tqdm
import torch
from ir_measures import ScoredDoc
from collections import defaultdict
import numpy as np


def simple_batching(iterable, batch_size=64):
    batch = []
    for i in iterable:
        batch.append(i)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def compute_batch_stats(query_id: str,
                        batch: list,
                        predictions: dict,
                        run: list,
                        scores: list,
                        tokenized: dict,
                        tokenizer: AutoTokenizer,
                        block_to_analyze: list):
    batch_res = defaultdict(dict)
    for batch_id, ((doc_id, _), score, input_ids) in enumerate(zip(batch, scores, tokenized["input_ids"])):
        run.append(ScoredDoc(query_id, doc_id, score))
        batch_res[doc_id]["score"] = score
        sep_pos = (input_ids == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]

        # Find token positions
        end_query = sep_pos[0].item()
        end_doc = sep_pos[-1].item()
        query_tok_pos = np.array(range(1, end_query))
        doc_tok_pos = np.array(range(end_query + 1, end_doc))
        spec_tok_pos = np.array([0, end_query, end_doc])
        batch_res[doc_id]["query_tokens_pos"] = torch.tensor(query_tok_pos)
        batch_res[doc_id]["doc_tokens_pos"] = torch.tensor(doc_tok_pos)
        batch_res[doc_id]["spec_tokens_pos"] = torch.tensor(spec_tok_pos)

        # Compute distances
        batch_res[doc_id]["dists"] = []
        for i, transformer_block in enumerate(predictions["hidden_states"]):
            if i not in block_to_analyze:
                continue
            # compute cosine similarity for each layer
            layer_embds = transformer_block[batch_id]
            norms_prod = torch.norm(layer_embds, dim=1).reshape(-1, 1) @ torch.norm(layer_embds, dim=1).reshape(1, -1)
            dist_matrix = (layer_embds @ layer_embds.T) / norms_prod

            batch_res[doc_id]["dists"].append(dist_matrix)
    return batch_res


def main():
    parser = argparse.ArgumentParser(prog="precompute distances for NEED framework")
    parser.add_argument("--model", type=str, default="veneres/monobert")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="msmarco-passage/trec-dl-2019/judged")
    parser.add_argument("--output_folder", type=str, default="/data")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--start-block", type=int, default=0)
    parser.add_argument("--end-block", type=int, default=13)
    parser.add_argument("--step-block", type=int, default=1)

    args = parser.parse_args()

    model = args.model
    device = args.device
    dataset = args.dataset
    output_folder = args.output_folder
    batch_size = args.batch_size
    start_block = args.start_block
    end_block = args.end_block
    step_block = args.step_block
    block_to_analyze = range(start_block, end_block, step_block)

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(model, output_hidden_states=True)
    model = model.to(device)

    model.eval()

    dataset = ir_datasets.load(dataset)
    docstore = dataset.docs_store()
    query_store = {query.query_id: query.text for query in dataset.queries_iter()}

    run = []

    scoreddoc_by_query = defaultdict(list)
    for query_id, doc_id, score in dataset.scoreddocs_iter():
        scoreddoc_by_query[query_id].append((doc_id, score))

    with torch.no_grad():
        for query_id, scoreddocs in scoreddoc_by_query.items():
            print(f"Processing query {query_id}")
            n_batches = math.ceil(len(scoreddocs) / batch_size)
            query_res_to_save = {}
            for batch_n, batch in tqdm(enumerate(simple_batching(scoreddocs, batch_size)), total=n_batches):
                batch_queries = [query_store[query_id]] * len(batch)
                batch_docs = [docstore.get(doc_id).text for doc_id, score in batch]
                tokenized = tokenizer(batch_queries, batch_docs, return_tensors="pt", padding="max_length",
                                      truncation=True)

                tokenized = tokenized.to(device)
                predictions = model(**tokenized)

                scores = torch.softmax(predictions.logits, dim=1)[:, 1].detach().cpu().numpy()

                batch_res = compute_batch_stats(query_id,
                                                batch,
                                                predictions,
                                                run,
                                                scores,
                                                tokenized,
                                                tokenizer,
                                                list(block_to_analyze))

                query_res_to_save = {**query_res_to_save, **batch_res}

            torch.save(query_res_to_save, f"{output_folder}/{query_id}.pt")


if __name__ == '__main__':
    main()
