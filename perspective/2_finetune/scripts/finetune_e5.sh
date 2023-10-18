epoch=5
scale=small
for dataset in few_rel_nat
do
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python finetune_e5.py \
        --model_name_or_path intfloat/e5-large \
        --output_dir checkpoints/finetune-pretrain-1024-gpt-noprior-compare/e5-large-${dataset}-epoch=${epoch} \
        --train_file ../converted_triplet_results/few_rel_nat_embed=e5_n=1024_m=1024_d=77.0_choice_seed=100-gpt-3.5-turbo-train.json \
        --cache_dir cache \
        --max_source_length 512 \
        --num_train_epochs $epoch \
        --per_device_train_batch_size 4 \
        --learning_rate 2e-6 \
        --save_steps 1280 \
        --cl_temperature 0.01
done