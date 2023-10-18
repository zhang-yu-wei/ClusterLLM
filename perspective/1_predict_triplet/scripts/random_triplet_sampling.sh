scale=small
for dataset in banking77
do
    for max_query in 1024
    do
        python random_triplet_sampling.py \
            --data_path ../../datasets/${dataset}/${scale}.jsonl \
            --dataset $dataset \
            --scale $scale \
            --max_query $max_query \
            --out_dir sampled_triplet_results \
            --seed 100
    done
done