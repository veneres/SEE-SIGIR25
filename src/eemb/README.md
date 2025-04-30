# Earlyexiting monobert

Copy-pasted from https://github.com/castorini/earlyexiting-monobert/ to ensure reproducibility.

The only difference is in the output of the model, changed for compatibility with the rest of the code which are 
higlighted below using the output of the following `diff` command:

```bash
diff -bur --color src/eemb earlyexiting-monobert/transformers
```

where `earlyexiting-monobert/transformers` is the folder containing the original code directly copied from the repository.

```bash
diff --color -bur --color src/eemb/data/metrics/__init__.py earlyexiting-monobert/transformers/data/metrics/__init__.py
--- src/eemb/data/metrics/__init__.py	2024-04-09 11:15:37
+++ earlyexiting-monobert/transformers/data/metrics/__init__.py	2025-04-29 15:21:54
@@ -65,9 +65,7 @@
             return {"acc": simple_accuracy(preds, labels)}
         elif task_name == "asnq":
             return {"acc": simple_accuracy(preds, labels)}
-        elif task_name == "trec-dl-2019":
-            return {"acc": simple_accuracy(preds, labels)}
-        elif task_name == "trec-dl-2020":
+        elif task_name == "trec-dl":
             return {"acc": simple_accuracy(preds, labels)}
         elif task_name == "msmarco":
             return acc_and_f1(preds, labels)
```

```bash
diff --color -bur --color src/eemb/data/processors/glue.py earlyexiting-monobert/transformers/data/processors/glue.py
--- src/eemb/data/processors/glue.py	2024-04-09 11:15:37
+++ earlyexiting-monobert/transformers/data/processors/glue.py	2025-04-29 15:21:54
@@ -840,8 +840,7 @@
     "sick": SickProcessor,
     "msmarco": MsmarcoProcessor,
     "asnq": AsnqProcessor,
-    "trec-dl-2019": TrecdlProcessor,
-    "trec-dl-2020": TrecdlProcessor,
+    "trec-dl": TrecdlProcessor,
 }

 glue_output_modes = {
@@ -858,6 +857,5 @@
     "sick": "regression",
     "msmarco": "classification",
     "asnq": "classification",
-    "trec-dl-2019": "classification",
-    "trec-dl-2020": "classification",
+    "trec-dl": "classification",
 }
 
```

```bash
diff --color -bur --color src/eemb/modeling_highway_bert.py earlyexiting-monobert/transformers/modeling_highway_bert.py
--- src/eemb/modeling_highway_bert.py	2024-04-09 11:15:37
+++ earlyexiting-monobert/transformers/modeling_highway_bert.py	2025-04-29 15:21:54
@@ -461,7 +461,7 @@
                 raise NotImplementedError("Wrong training strategy!")

         if not self.training:
-            outputs = {"logits": logits}
+            outputs = outputs + ((original_entropy, highway_entropy), exit_layer)
             if output_layer >= 0:
                 outputs = (outputs[0],) +\
                           (highway_all_logits[output_layer],) +\
```

```bash
diff --color -bur --color src/eemb/tokenization_bert.py earlyexiting-monobert/transformers/tokenization_bert.py
--- src/eemb/tokenization_bert.py	2024-04-09 11:15:37
+++ earlyexiting-monobert/transformers/tokenization_bert.py	2025-04-29 15:21:54
@@ -22,8 +22,6 @@
 import unicodedata
 from io import open

-import torch
-
 from .tokenization_utils import PreTrainedTokenizer

 logger = logging.getLogger(__name__)
@@ -266,49 +264,6 @@
                 writer.write(token + u'\n')
                 index += 1
         return (vocab_file,)
-
-    def __call__(self, batch_queries, batch_queries_docs, **kwargs):
-        pad_on_left = False
-        max_length = 512
-        pad_token = 0
-        mask_padding_with_zero = True
-        pad_token_segment_id = 0
-        label_list = ['0', '1']
-
-        if len(batch_queries) > 1 or len(batch_queries_docs) > 1:
-            raise ValueError("Batching is not supported for this tokenizer, see original code for more details.")
-
-        inputs = self.encode_plus(batch_queries[0],
-                                  batch_queries_docs[0],
-                                  add_special_tokens=True,
-                                  max_length=max_length
-                                  )
-        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
-
-        # The mask has 1 for real tokens and 0 for padding tokens. Only real
-        # tokens are attended to.
-        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
-
-        # Zero-pad up to the sequence length.
-        padding_length = max_length - len(input_ids)
-        if pad_on_left:
-            input_ids = ([pad_token] * padding_length) + input_ids
-            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
-            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
-        else:
-            input_ids = input_ids + ([pad_token] * padding_length)
-            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
-            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
-
-        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
-        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
-                                                                                            max_length)
-        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
-                                                                                            max_length)
-        return {"input_ids": torch.tensor(input_ids).reshape(1, -1),
-                "attention_mask": torch.tensor(attention_mask).reshape(1, -1),
-                "token_type_ids": torch.tensor(token_type_ids).reshape(1, -1)
-                }
```