prompt_path=prompts_pair_exps_pair_v8.json
for dataset in banking77
do
    sampled_pair_path=sampled_pair_results/${dataset}_embed=finetuned_s=small_k=1_multigran2-200_seed=100.json
    data_path=../datasets/${dataset}/small.jsonl
    python sample_pairs_for_prompt.py \
        --prompt_path $prompt_path \
        --sampled_pair_path $sampled_pair_path \
        --data_path $data_path \
        --dataset $dataset \
        --seed 1234
done
