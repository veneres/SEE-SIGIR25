#!/bin/bash

# echo mode on
set -x

data_path=$1
batch_size=$2


# Preprocess data and create a sample of 100 documents from the training set of msmarco-passage/train/judged

python src/00_rq1_prepoc_train_subset.py

# Compute scores

####################################################################################################################
# Please note that in the paper we reported the results only for veneres/monobert, so you can avoid running the other
# models if you want to save time. The other models are included for completeness also in the data folder.
#####################################################################################################################
# Evaluation on training set sample for msmarco

for model in 'veneres/monobert' 'veneres/monoelectra' 'cross-encoder/ms-marco-MiniLM-L-12-v2'; do
    python src/00_rq1_compute.py --model_name=$model --exit_level=13 --batch_size="$batch_size" --device=cuda --output_dir="$data_path"/msmarco-train-sample/$model --scoreddoc_file="/data/rq1/sample_scoreddocs_train.csv"
done

# Compute the similarity scores for the training set sample

for sim_type in 'amax' 'max' 'mean' 'mean_centroid'; do
    for model in 'veneres/monobert' 'veneres/monoelectra' 'cross-encoder/ms-marco-MiniLM-L-12-v2'; do
      python src/00_rq1_eval.py --data_path="$data_path"/msmarco-train-sample/$model/$sim_type
    done
done
