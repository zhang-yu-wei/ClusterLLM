# randomly sample triplets
import os
import json
import random
import argparse
import numpy as np

def load_data(args):
    # data_path = f"/data3/yuwei/proj_weaksupervise/code/evalMetaICL/unlabeled_data/{args.dataset}/{args.dataset}_{args.num_data}_{args.seed}_{args.split}.jsonl"
    return [json.loads(l) for l in open(args.data_path, 'r')]


def generate(args):
    os.makedirs(args.out_dir, exist_ok=True)
    save_path = f"{args.out_dir}/{args.dataset}_s={args.scale}_m={args.max_query}_random_seed={args.seed}.json"
    print(save_path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    data = load_data(args)
    inp = [d['input'] for d in data]
    labels = [d['label'] for d in data]

    triplets = []
    inds = list(range(len(data)))
    while len(triplets) < args.max_query:
        query, choice1, choice2 = random.sample(inds, 3)
        if (query, choice1, choice2) not in triplets \
            and choice1 != query and choice2 != query:
            triplets.append((query, choice1, choice2))

    result = []
    for trip in triplets:
        output = None
        if random.random() > 0.5:
            input_txt = "Query: " + inp[trip[0]] + "\nChoice 1: " + inp[trip[1]] + "\nChoice 2: " + inp[trip[2]] + "\nChoice"
            # for analyzing purpose
            if (labels[trip[0]] == labels[trip[1]]) and \
            (labels[trip[0]] != labels[trip[2]]):
                output = " 1"
            elif (labels[trip[0]] != labels[trip[1]]) and \
                (labels[trip[0]] == labels[trip[2]]):
                output = " 2"
            elif (labels[trip[0]] == labels[trip[1]]) and \
                (labels[trip[0]] == labels[trip[2]]):
                output = "both"
            result.append({
                "input": input_txt,
                "output": output,
                "options": [" 1", " 2"],
                "task": args.dataset,
                "query_idx": int(trip[0]),
                "choice1_idx": int(trip[1]),
                "choice2_idx": int(trip[2]),
            })
        else:
            input_txt = "Query: " + inp[trip[0]] + "\nChoice 1: " + inp[trip[2]] + "\nChoice 2: " + inp[trip[1]] + "\nChoice"
            # for analyzing purpose
            if (labels[trip[0]] == labels[trip[1]]) and \
            (labels[trip[0]] != labels[trip[2]]):
                output = " 2"
            elif (labels[trip[0]] != labels[trip[1]]) and \
                (labels[trip[0]] == labels[trip[2]]):
                output = " 1"
            elif (labels[trip[0]] == labels[trip[1]]) and \
                (labels[trip[0]] == labels[trip[2]]):
                output = "both"
            result.append({
                "input": input_txt,
                "output": output,
                "options": [" 1", " 2"],
                "task": args.dataset,
                "query_idx": int(trip[0]),
                "choice1_idx": int(trip[2]),
                "choice2_idx": int(trip[1]),
            })
    
    print("#GT", len([res for res in result if res['output'] not in ['both', None]]))
    with open(save_path, 'w') as f:
        json.dump(result, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--scale", type=str, default="small")
    parser.add_argument("--max_query", type=int, default=256)
    parser.add_argument("--out_dir", default="links", type=str)
    parser.add_argument("--seed", type=int, default=100)

    args = parser.parse_args()
    generate(args)