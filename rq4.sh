
#echo on
set -x

batch_size=$1
device=$2
eemb_model_msmarco=$3
base_output_path=$4
dataset_name=$5
n_reps=$6
lt=$7 # list truncation

see_fixed_exit_param=0.3
eemb_fixed_param=0.95

model="veneres/monobert"

# SEE evaluation

python src/02_speed_test_full.py \
    --model=$model \
    --dataset="$dataset_name"\
    --output="$base_output_path"/"$dataset_name"/"$model"_full_"$lt".csv \
    --batch_size="$batch_size" \
    --n_reps="$n_reps" \
    --device="$device" \
    --eval_metric=ndcg \
    --list_truncation="$lt"

python src/02_speed_test_see.py \
  --model=$model \
  --dataset="$dataset_name" \
  --output="$base_output_path"/"$dataset_name"/"$model"_see_"$lt".csv \
  --batch_size="$batch_size" \
  --n_reps="$n_reps" \
  --exit_level=0 \
  --exit_mode=2 \
  --exit_param=$see_fixed_exit_param \
  --device="$device" \
  --exit_sim=amax \
  --eval_metric=ndcg \
  --list_truncation="$lt"

# eeMB+ with batch_size=batch_size and eemb original

python src/02_speed_test_eemb.py \
     --model_path="$eemb_model_msmarco" \
     --dataset="$dataset_name" \
     --output="$base_output_path"/"$dataset_name"/eemb_plus_"$lt".csv \
     --n_reps="$n_reps" \
     --device="$device" \
     --batch_size="$batch_size" \
     --nc=$eemb_fixed_param \
     --eval_metric=ndcg \
     --list_truncation="$lt"
