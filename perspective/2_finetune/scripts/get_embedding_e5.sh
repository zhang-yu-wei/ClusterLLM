# ===== original embedding =====
for dataset in banking77
do
    for scale in small
    do
        CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python get_embedding_e5.py \
            --model-name-or-path intfloat/e5-large \
            --input_path ../../datasets/${dataset}/${scale}.jsonl \
            --output_path ../../datasets/${dataset}/${scale}_embeds_e5.hdf5 \
            --scale $scale \
            --measure
    done
done

# ===== with checkpoint =====
# dataset=banking77
# scale=small
# checkpoint_path=CHECKPOINT_PATH
# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python get_embedding_e5.py \
#     --model-name-or-path intfloat/e5-large \
#     --checkpoint $checkpoint_path \
#     --input_path ../../datasets/${dataset}/${scale}.jsonl \
#     --output_path ${checkpoint}/${scale}_embeds_e5.hdf5 \
#     --scale $scale \
#     --measure