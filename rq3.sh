
#echo on
set -x

batch_size=$1
device=$2
model=$3
eemb_model=$4
base_output_path=$5
dataset_name=$6
n_reps=$7
eval_metric=$8



# Speed test for the model without early exit
python src/02_speed_test_full.py \
    --model="$model" \
    --dataset="$dataset_name"\
    --output="$base_output_path"/"$dataset_name"/full.csv \
    --batch_size="$batch_size" \
    --n_reps="$n_reps" \
    --device="$device" \
    --eval_metric="$eval_metric"

# Speed test for SEE varying exit_param
for exitparam in 0.2 0.3 0.4 0.5 0.9;
  do
    python src/02_speed_test_see.py \
    --model="$model" \
    --dataset="$dataset_name" \
    --output="$base_output_path"/"$dataset_name"/see_"$exitparam".csv \
    --batch_size="$batch_size" \
    --n_reps="$n_reps" \
    --exit_level=0 \
    --exit_mode=2 \
    --exit_param=$exitparam \
    --device="$device" \
    --exit_sim=amax \
    --eval_metric="$eval_metric"
done



# eeMB+ with batch_size=batch_size and varying exit_param
for exitparam in 0.7 0.8 0.9 0.95 1;
do
  python src/02_speed_test_eemb.py \
       --model_path="$eemb_model" \
       --dataset=$dataset_name \
       --output="$base_output_path"/"$dataset_name"/eemb_plus_"$exitparam".csv \
       --n_reps="$n_reps" \
       --device="$device" \
       --batch_size="$batch_size" \
       --nc=$exitparam \
       --eval_metric="$eval_metric"
done


# eeMB original implementation with batch_size=1 and varying exit_param
for exitparam in 0.7 0.8 0.9 0.95 1;
do
  python src/02_speed_test_full.py \
     --model_path="$eemb_model" \
     --dataset="$dataset_name" \
     --output="$base_output_path"/"$dataset_name"/eemb_"$exitparam".csv \
     --batch_size=1 \
     --n_reps="$n_reps" \
     --device="$device" \
     --batch_size=1 \
     --nc="$exitparam" \
     --eval_metric="$eval_metric"
done