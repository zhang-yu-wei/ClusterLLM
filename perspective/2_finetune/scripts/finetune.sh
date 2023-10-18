epoch=15
scale=small
d="67.0"
seed=100
dataset=banking77

# ===== gpt =====
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python finetune.py \
    --model_name_or_path hkunlp/instructor-large \
    --output_dir checkpoints/finetune-pretrain-1024-gpt-noprior/instructor-large-${dataset}-d=${d}-epoch=${epoch} \
    --train_file converted_triplet_results/${dataset}_embed=instructor_s=${scale}_m=1024_d=${d}_sf_choice_seed=${seed}-gpt-3.5-turbo-0301-train.json \
    --cache_dir cache \
    --max_source_length 512 \
    --num_train_epochs $epoch \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-6 \
    --save_steps 3840 \
    --cl_temperature 0.01 \
    --overwrite_output_dir

# ===== self =====
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python finetune.py \
    --model_name_or_path hkunlp/instructor-large \
    --output_dir checkpoints/finetune-pretrain-1024-self-noprior/instructor-large-${dataset}-d=${d}-epoch=${epoch} \
    --train_file converted_triplet_results/${dataset}_embed=instructor_s=${scale}_m=1024_d=${d}_sf_choice_seed=100-self-train.json \
    --cache_dir cache \
    --max_source_length 512 \
    --num_train_epochs $epoch \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-6 \
    --save_steps 3840 \
    --cl_temperature 0.01 \
    --overwrite_output_dir