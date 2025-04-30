from copy import deepcopy
from pathlib import Path
from typing import Tuple

import torch

from transformers import BertForSequenceClassification, BertTokenizerFast, AutoTokenizer, \
    AutoModelForSequenceClassification, RobertaForSequenceClassification, ElectraForSequenceClassification, \
    RobertaTokenizerFast, ElectraTokenizerFast
from transformers.utils import logging

from eemb import BertConfig
from eemb.modeling_highway_bert import BertForSequenceClassification as eeMB

logger = logging.get_logger(__name__)


def get_extended_attention_mask(attention_mask: torch.Tensor,
                                input_shape: Tuple[int],
                                dtype: torch.float
                                ) -> torch.Tensor:
    """
    Copied and modified from transformers.modeling_utils.py

    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


class EEMBSlice(torch.nn.Module):

    def __init__(self,
                 layer: torch.nn,
                 classifier: torch.nn,
                 embeddings: torch.nn,
                 sep_token_id: int,
                 pad_token_id: int,
                 tokenizer: AutoTokenizer,
                 is_first_slice: bool = False,
                 is_last_slice: bool = False,
                 pooler: torch.nn = None,
                 dropout: torch.nn = None):
        super().__init__()
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.embeddings = embeddings
        self.layer = layer
        self.classifier = classifier
        self.is_first_slice = is_first_slice
        self.is_last_slice = is_last_slice
        if is_last_slice and is_first_slice:
            raise ValueError("is_last_slice and is_first_slice cannot be both True")

        if pooler is None and is_last_slice:
            raise ValueError("pooler must be provided if is_last_slice is True")

        if dropout is None and is_last_slice:
            raise ValueError("dropout must be provided if is_last_slice is True")

        self.pooler = pooler

        self.dropout = dropout

        self.tokenizer = tokenizer

    def forward(self,
                embeddings: torch.Tensor = None,
                input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                token_type_ids: torch.Tensor = None):

        if embeddings is None and not self.is_first_slice:
            raise ValueError("embeddings must be provided if is_first_slice is True")

        if self.is_first_slice:
            embeddings = self.embeddings(input_ids, token_type_ids)

            input_shape = input_ids.size()

            attention_mask = get_extended_attention_mask(attention_mask, input_shape, dtype=embeddings.dtype)

        layer_embeddings = self.layer(
            embeddings,
            attention_mask=attention_mask,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
        )

        if self.is_last_slice:
            sequence_output = layer_embeddings[0]
            pooled_output = self.pooler(sequence_output)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        else:
            logits = self.classifier(layer_embeddings)[0]

        probs = torch.softmax(logits, dim=1)
        return layer_embeddings[0], attention_mask, probs

    @staticmethod
    def from_pretrained(model_path: str, device):

        model_path = Path(model_path)
        config = BertConfig.from_pretrained(model_path / "config.json", num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = eeMB.from_pretrained(model_path, config=config)
        model = model.to(device)
        model.eval()
        slices = []

        for layer_n in range(12):
            slice = EEMBSlice(model.bert.encoder.layer[layer_n],
                              model.bert.encoder.highway[layer_n] if layer_n < 11 else model.classifier,
                              model.bert.embeddings,
                              tokenizer.sep_token_id,
                              tokenizer.pad_token_id,
                              tokenizer,
                              is_first_slice=layer_n == 0,
                              is_last_slice=layer_n == 11,
                              pooler=model.bert.pooler,
                              dropout=model.dropout)
            slices.append(slice)
        return slices, tokenizer


class AutoModelFirstSliced(torch.nn.Module):

    def __init__(self,
                 embeddings: torch.nn,
                 sliced_encoder: torch.nn,
                 sim_function: str,
                 sep_token_id: int,
                 pad_token_id: int,
                 tokenizer: AutoTokenizer,
                 should_explain: bool = False):
        super().__init__()
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.embeddings = embeddings
        self.sliced_encoder = sliced_encoder
        self._sim_function_mapping = {"amax": lambda x: torch.amax(x, dim=2).sum(dim=1),
                                      "max": lambda x: torch.max(torch.amax(x, dim=2), dim=1).values,
                                      "mean": lambda x: x.sum(dim=[1, 2]) / torch.count_nonzero(x, dim=[1, 2]),
                                      "mean_centroid": None}

        self._placeholder_value = 0
        if sim_function is not None and sim_function not in self._sim_function_mapping.keys():
            raise ValueError(f"sim_function must be among {list(self._sim_function_mapping.keys())}")
        self.sim_function = sim_function
        self.should_explain = should_explain
        self.tokenizer = tokenizer

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor,
                all_similarities: bool = False,
                skip_similarity_computation: bool = False):

        embeddings = self.embeddings(input_ids, token_type_ids)

        input_shape = input_ids.size()

        attention_mask = get_extended_attention_mask(attention_mask, input_shape, dtype=embeddings.dtype)
        encoder_outputs = self.sliced_encoder(
            embeddings,
            attention_mask=attention_mask,
            head_mask=[None] * len(self.sliced_encoder.layer),
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=all_similarities,
            return_dict=True,
        )
        # change token_type_ids if all zeros (as in roberta)
        if torch.count_nonzero(token_type_ids) == 0:
            values, indices = torch.max((input_ids == self.sep_token_id), dim=1)
            # reshape to (batch_size, 1)
            indices = indices.reshape(-1, 1)
            indices = indices.to(embeddings.device)
            # reshape to (token_type_ids.shape[0], token_type_ids.shape[1])
            mask_arange = torch.arange(0, token_type_ids.shape[1]).repeat(token_type_ids.shape[0], 1)
            mask_arange = mask_arange.to(embeddings.device)
            doc_indices = mask_arange > indices
            token_type_ids = torch.where(doc_indices, 1, 0)
            token_type_ids[input_ids == self.pad_token_id] = 0

        query_token_pos = (input_ids > 0).logical_and(token_type_ids == 0)
        query_token_pos[:, 0] = False  # Remove CLS token
        query_token_pos = query_token_pos.logical_and(input_ids != self.sep_token_id)  # Remove SEP token
        doc_token_pos = (input_ids > 0).logical_and(token_type_ids == 1)
        doc_token_pos = doc_token_pos.logical_and(input_ids != self.sep_token_id)  # Remove SEP token
        token_contributions = {}
        # slow. only for debugging, make sure to set should_explain to False in production. Only amax is supported
        if self.should_explain:
            token_contributions = self._explain(input_ids,
                                                attention_mask,
                                                token_type_ids,
                                                embeddings,
                                                doc_token_pos,
                                                query_token_pos)

        if not all_similarities:
            embeddings = encoder_outputs[0]
            sim_matrix = None
            if not skip_similarity_computation:
                sim_matrix = self._compute_sim(doc_token_pos, embeddings, query_token_pos)
            return {"similarities": sim_matrix,
                    "hidden_states": embeddings,
                    "attention_mask": attention_mask,
                    "contributions": token_contributions
                    }
        else:
            res = []
            for i, embeddings in enumerate(encoder_outputs["hidden_states"]):

                if self.sim_function is not None:
                    sim_matrix = self._compute_sim(doc_token_pos, embeddings, query_token_pos)
                else:
                    sim_matrix = {}
                    for sim_function in ["max", "amax", "mean", "mean_centroid"]:
                        self.sim_function = sim_function
                        sim_matrix[sim_function] = self._compute_sim(doc_token_pos, embeddings, query_token_pos)
                        self.sim_function = None

                res.append({"similarities": sim_matrix,
                            "hidden_states": embeddings,
                            "attention_mask": attention_mask,
                            "contributions": token_contributions})
            return res

    def _compute_sim(self,
                     doc_token_pos: torch.Tensor,
                     embeddings: torch.Tensor,
                     query_token_pos: torch.Tensor) -> torch.Tensor:
        if self.sim_function == "mean_centroid":
            queries_only = torch.clone(embeddings)
            queries_only[torch.logical_not(query_token_pos)] = 0

            docs_only = torch.clone(embeddings)
            docs_only[torch.logical_not(doc_token_pos)] = 0

            mean_query = torch.sum(queries_only, dim=1) / torch.count_nonzero(queries_only, dim=1)
            mean_doc = torch.sum(docs_only, dim=1) / torch.count_nonzero(docs_only, dim=1)
            sim_matrix = torch.bmm(mean_query.reshape(mean_query.shape[0], 1, -1),
                                   mean_doc.reshape(mean_doc.shape[0], -1, 1))
            return torch.flatten(sim_matrix.reshape(sim_matrix.shape[0], -1))

        batch_size = embeddings.shape[0]
        norms_prod = torch.bmm(torch.norm(embeddings, dim=2).reshape(batch_size, -1, 1),
                               torch.norm(embeddings, dim=2).reshape(batch_size, 1, -1))
        sim_matrix = torch.bmm(embeddings, embeddings.transpose(1, 2)) / norms_prod

        sim_matrix[torch.logical_not(query_token_pos)] = self._placeholder_value
        sim_matrix = sim_matrix.transpose(1, 2)
        sim_matrix[torch.logical_not(doc_token_pos)] = self._placeholder_value
        sim_matrix = sim_matrix.transpose(1, 2)
        mask = sim_matrix != 0
        sim_matrix = (sim_matrix * mask)

        return self._sim_function_mapping[self.sim_function](sim_matrix)

    def _compute_matrix(self,
                        doc_token_pos: torch.Tensor,
                        embeddings: torch.Tensor,
                        query_token_pos: torch.Tensor) -> torch.Tensor:
        if self.sim_function == "mean_centroid":
            queries_only = torch.clone(embeddings)
            queries_only[torch.logical_not(query_token_pos)] = 0

            docs_only = torch.clone(embeddings)
            docs_only[torch.logical_not(doc_token_pos)] = 0

            mean_query = torch.sum(queries_only, dim=1) / torch.count_nonzero(queries_only, dim=1)
            mean_doc = torch.sum(docs_only, dim=1) / torch.count_nonzero(docs_only, dim=1)
            sim_matrix = torch.bmm(mean_query.reshape(mean_query.shape[0], 1, -1),
                                   mean_doc.reshape(mean_doc.shape[0], -1, 1))
            return torch.flatten(sim_matrix.reshape(sim_matrix.shape[0], -1))

        batch_size = embeddings.shape[0]
        norms_prod = torch.bmm(torch.norm(embeddings, dim=2).reshape(batch_size, -1, 1),
                               torch.norm(embeddings, dim=2).reshape(batch_size, 1, -1))
        sim_matrix = torch.bmm(embeddings, embeddings.transpose(1, 2)) / norms_prod

        sim_matrix[torch.logical_not(query_token_pos)] = self._placeholder_value
        sim_matrix = sim_matrix.transpose(1, 2)
        sim_matrix[torch.logical_not(doc_token_pos)] = self._placeholder_value
        sim_matrix = sim_matrix.transpose(1, 2)
        mask = sim_matrix != 0
        sim_matrix = (sim_matrix * mask)
        return sim_matrix

    def _explain(self, input_ids, attention_mask, token_type_ids, embeddings, doc_token_pos, query_token_pos):

        sim_matrix = self._compute_matrix(doc_token_pos, embeddings, query_token_pos)
        contributors = torch.argmax(sim_matrix, dim=2)
        contributions = torch.amax(sim_matrix, dim=2)
        contrib_dict = {'contributors': [], 'contribution_values': []}
        for i in range(contributors.shape[0]):
            pair_contributors = contributors[i]
            pair_input_ids = input_ids[i]
            pair_contributions = contributions[i]
            ids_of_contributors = pair_input_ids[pair_contributors]
            contrib_dict['contributors'].append(
                self.tokenizer.convert_ids_to_tokens(ids_of_contributors, skip_special_tokens=True))
            contrib_dict['contribution_values'].append(pair_contributions[torch.nonzero(pair_contributions)].squeeze())
        return contrib_dict

    def string(self):
        return str(self.embeddings)

    def to(self, *args, **kwargs):
        self.embeddings = self.embeddings.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    @staticmethod
    def from_pretrained(model: AutoModelForSequenceClassification,
                        tokenizer: AutoTokenizer,
                        end_layer: int,
                        sim_function="amax", should_explain: bool = False):
        if type(model) is BertForSequenceClassification:
            return BertFirstSliced(model, tokenizer, end_layer=end_layer, sim_function=sim_function,
                                   should_explain=should_explain)
        elif type(model) is RobertaForSequenceClassification:
            return RobertaFirstSliced(model, tokenizer, end_layer=end_layer, sim_function=sim_function,
                                      should_explain=should_explain)
        elif type(model) is ElectraForSequenceClassification:
            return ElectraFirstSliced(model, tokenizer, end_layer=end_layer, sim_function=sim_function,
                                      should_explain=should_explain)
        else:
            raise ValueError("Model not supported")


class BertFirstSliced(AutoModelFirstSliced):
    def __init__(self,
                 bert_model: BertForSequenceClassification,
                 tokenizer: BertTokenizerFast,
                 end_layer: int,
                 sim_function: str = "amax", should_explain: bool = False):
        embeddings = bert_model.bert.embeddings
        sliced_encoder = deepcopy(bert_model.bert.encoder)
        sliced_encoder.layer = sliced_encoder.layer[:end_layer]
        super().__init__(embeddings=embeddings,
                         sim_function=sim_function,
                         should_explain=should_explain,
                         sep_token_id=tokenizer.sep_token_id,
                         pad_token_id=tokenizer.pad_token_id,
                         tokenizer=tokenizer,
                         sliced_encoder=sliced_encoder)


class RobertaFirstSliced(AutoModelFirstSliced):
    def __init__(self,
                 roberta_model: RobertaForSequenceClassification,
                 tokenizer: RobertaTokenizerFast,
                 end_layer: int,
                 sim_function: str = "amax", should_explain: bool = False):
        embeddings = roberta_model.roberta.embeddings
        sliced_encoder = deepcopy(roberta_model.roberta.encoder)
        sliced_encoder.layer = sliced_encoder.layer[:end_layer]

        super().__init__(embeddings=embeddings,
                         sim_function=sim_function,
                         should_explain=should_explain,
                         sep_token_id=tokenizer.sep_token_id,
                         pad_token_id=tokenizer.pad_token_id,
                         tokenizer=tokenizer,
                         sliced_encoder=sliced_encoder)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor = None,
                all_similarities: bool = False):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        return super().forward(input_ids, attention_mask, token_type_ids, all_similarities)


class ElectraFirstSliced(AutoModelFirstSliced):
    def __init__(self,
                 electra_model: ElectraForSequenceClassification,
                 tokenizer: ElectraTokenizerFast,
                 end_layer: int,
                 sim_function: str = "amax", should_explain: bool = False):
        embeddings = electra_model.electra.embeddings
        sliced_encoder = deepcopy(electra_model.electra.encoder)
        sliced_encoder.layer = sliced_encoder.layer[:end_layer]

        super().__init__(embeddings=embeddings,
                         sim_function=sim_function,
                         should_explain=should_explain,
                         tokenizer=tokenizer,
                         sep_token_id=tokenizer.sep_token_id,
                         pad_token_id=tokenizer.pad_token_id,
                         sliced_encoder=sliced_encoder)


class AutoModelLastSliced(torch.nn.Module):

    def __init__(self, embeddings: torch.nn, sliced_encoder: torch.nn, pooler: torch.nn, classifier: torch.nn):

        super().__init__()
        self.embeddings = embeddings
        self.sliced_encoder = sliced_encoder
        self.pooler = pooler
        self.classifier = classifier

    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor):
        encoder_outputs = self.sliced_encoder(
            embeddings,
            attention_mask=attention_mask,
            head_mask=[None] * len(self.sliced_encoder.layer),
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        pooled_output = pooled_output

        logits = self.classifier(pooled_output)

        if logits.shape[1] == 1:  # cross-encoder/ms-marco-MiniLM-L-12-v2 outputs only one values (relevance)
            scores = torch.sigmoid(logits).ravel()
        else:
            scores = torch.softmax(logits, dim=1)[:, 1]

        return scores

    def string(self):
        return str(self.embeddings)

    def to(self, *args, **kwargs):
        self.embeddings = self.embeddings.to(*args, **kwargs)
        self.sliced_encoder = self.sliced_encoder.to(*args, **kwargs)
        if self.pooler is not None:
            self.pooler = self.pooler.to(*args, **kwargs)
        self.classifier = self.classifier.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    @staticmethod
    def from_pretrained(model: AutoModelForSequenceClassification, start_layer: int):
        if type(model) is BertForSequenceClassification:
            return BertLastSliced(model, start_layer=start_layer)
        elif type(model) is RobertaForSequenceClassification:
            return RobertaLastSliced(model, start_layer=start_layer)
        elif type(model) is ElectraForSequenceClassification:
            return ElectraLastSliced(model, start_layer=start_layer)
        else:
            raise ValueError("Model not supported")


class BertLastSliced(AutoModelLastSliced):
    def __init__(self, bert_model: BertForSequenceClassification, start_layer: int):
        embeddings = bert_model.bert.embeddings

        sliced_encoder = deepcopy(bert_model.bert.encoder)
        sliced_encoder.layer = sliced_encoder.layer[start_layer:]
        pooler = deepcopy(bert_model.bert.pooler)
        classifier = deepcopy(bert_model.classifier)

        super().__init__(embeddings=embeddings,
                         sliced_encoder=sliced_encoder,
                         pooler=pooler,
                         classifier=classifier)


class RobertaLastSliced(AutoModelLastSliced):
    def __init__(self, bert_model: RobertaForSequenceClassification, start_layer: int):
        embeddings = bert_model.roberta.embeddings

        sliced_encoder = deepcopy(bert_model.roberta.encoder)
        sliced_encoder.layer = sliced_encoder.layer[start_layer:]
        pooler = deepcopy(bert_model.roberta.pooler)
        classifier = deepcopy(bert_model.classifier)

        super().__init__(embeddings=embeddings,
                         sliced_encoder=sliced_encoder,
                         pooler=pooler,
                         classifier=classifier)

    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor):
        encoder_outputs = self.sliced_encoder(
            embeddings,
            attention_mask=attention_mask,
            head_mask=[None] * len(self.sliced_encoder.layer),
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        sequence_output = encoder_outputs[0]

        logits = self.classifier(sequence_output)
        scores = torch.softmax(logits, dim=1)[:, 1]

        return scores


class ElectraLastSliced(AutoModelLastSliced):
    def __init__(self,
                 electra_model: ElectraForSequenceClassification,
                 start_layer: int):
        embeddings = electra_model.electra.embeddings

        sliced_encoder = deepcopy(electra_model.electra.encoder)
        sliced_encoder.layer = sliced_encoder.layer[start_layer:]
        classifier = deepcopy(electra_model.classifier)

        super().__init__(embeddings=embeddings,
                         sliced_encoder=sliced_encoder,
                         pooler=None,
                         classifier=classifier)

    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor):
        encoder_outputs = self.sliced_encoder(
            embeddings,
            attention_mask=attention_mask,
            head_mask=[None] * len(self.sliced_encoder.layer),
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        sequence_output = encoder_outputs[0]

        logits = self.classifier(sequence_output)
        if logits.shape[1] == 1:
            scores = torch.sigmoid(logits).ravel()
        else:
            scores = torch.softmax(logits, dim=1)[:, 1]

        return scores

def main():
    model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dict_tokenizer = tokenizer(
        ["the most stable mineral at the earth's surface", "the most stable mineral at the earth's surface"],
        [
            "Most sandstone is composed of quartz or feldspar because they are the most resistant minerals to weathering processes at the Earth 's surface , as seen in Bowen 's reaction series .",
            "This weathering removed everything but quartz grains , the most stable mineral ."
        ],
        return_tensors="pt", padding="max_length", truncation=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.eval()
    model = model.to("cpu")
    first_slice = AutoModelFirstSliced.from_pretrained(model, tokenizer, end_layer=4, sim_function="amax",
                                                       should_explain=True)
    first_slice = first_slice.eval()
    res = first_slice(**dict_tokenizer, all_similarities=True)
    print(res["similarities"])
    print(res["contributions"])

    last_slice = AutoModelLastSliced.from_pretrained(model, start_layer=4)

    last_slice.eval()
    print(last_slice(embeddings=res["hidden_states"], attention_mask=res["attention_mask"]))

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.eval()
    model(**dict_tokenizer)

    print(torch.sigmoid(model(**dict_tokenizer).logits))


if __name__ == '__main__':
    main()
