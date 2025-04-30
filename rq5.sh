
#echo on
set -x

batch_size=$1
device=$2
base_output_path=$3
dataset_name=$4
n_reps=$5


see_fixed_exit_param=0.3
lt=1000

# SEE evaluation
for model in veneres/monoelectra cross-encoder/ms-marco-MiniLM-L-12-v2 naver/trecdl22-crossencoder-electra;
do

  python src/02_speed_test_full.py \
      --model=$model \
      --dataset="$dataset_name"\
      --output="$base_output_path"/"$dataset_name"/"$model"_full.csv \
      --batch_size="$batch_size" \
      --n_reps="$n_reps" \
      --device="$device" \
      --eval_metric=ndcg \
      --list_truncation="$lt"

  python src/02_speed_test_see.py \
    --model=$model \
    --dataset="$dataset_name" \
    --output="$base_output_path"/"$dataset_name"/"$model"_see.csv \
    --batch_size="$batch_size" \
    --n_reps="$n_reps" \
    --exit_level=0 \
    --exit_mode=2 \
    --exit_param=$see_fixed_exit_param \
    --device="$device" \
    --exit_sim=amax \
    --eval_metric=ndcg \
    --list_truncation="$lt"

done