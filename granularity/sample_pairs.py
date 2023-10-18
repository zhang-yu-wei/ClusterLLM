# instead of looking at just the closest
# we look at pairs in multiple granularities
import os
import json
import h5py
import argparse
import random
import numpy as np
from copy import deepcopy

from sklearn.cluster import AgglomerativeClustering

def load_data(args):
    # data_path = f"../datasets/{args.dataset}/{args.scale}.jsonl"
    return [json.loads(l) for l in open(args.data_path, 'r')]

def load_feat(args):
    feat_path = args.feat_path
    with h5py.File(feat_path, 'r') as f:
        X = f['embeds']
        X = np.asarray(X)
    return X

def generate(args):
    os.makedirs(args.out_dir, exist_ok=True)
    # save_path = f"{args.out_dir}/{args.dataset}_embed={args.embed_method}_n={args.num_data}_m={args.max_query}_multigran{args.min_clusters}-{args.max_clusters}_seed={args.seed}.json"
    save_path = f"{args.out_dir}/{args.dataset}_embed={args.embed_method}_s={args.scale}_k={args.k}_multigran{args.min_clusters}-{args.max_clusters}_seed={args.seed}.json"
    print(save_path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    data = load_data(args)
    inp = [d['input'] for d in data]
    # for analyzing purpose only
    labels = [d['label'] for d in data]
    X = load_feat(args)

    clustering = AgglomerativeClustering().fit(X)
    children = clustering.children_

    nodes = {idx: [idx] for idx in range(len(data))}
    cnt = len(data)
    clusters = []
    cur_clusters = list(range(len(data)))
    for child in children:
        nodes[cnt] = nodes[child[0]] + nodes[child[1]]

        cur_clusters.remove(child[0])
        cur_clusters.remove(child[1])
        cur_clusters.append(cnt)

        clusters.append(deepcopy(cur_clusters))
        cnt += 1
    
    # rerank examples according to distance to cluster center
    for idx in nodes:
        if len(nodes[idx]) <= 2:
            continue
        cur_embeds = X[nodes[idx]]
        cur_center = np.mean(cur_embeds, axis=0)
        cur_dists = ((cur_embeds - cur_center[None, :]) ** 2).sum(axis=-1)
        nodes[idx] = [nodes[idx][i] for i in np.argsort(cur_dists)]
    
    all_test_pairs = []
    # while len(all_test_pairs) < args.max_query:
    for _ in range(args.k):
        # order = list(range(args.min_clusters, args.max_clusters))
        # random.shuffle(order)
        # for step in order:
        for step in range(args.min_clusters, args.max_clusters):
            cls_idx1, cls_idx2 = children[-step] # look for the child before current step
            cls1 = nodes[cls_idx1]
            cls2 = nodes[cls_idx2]
            idx1 = random.choice(cls1)
            idx2 = random.choice(cls2)
            # p = (idx1, idx2)
            p = tuple(sorted([idx1, idx2]))
            if p not in all_test_pairs:
                all_test_pairs.append((p, step))
                # if len(all_test_pairs) >= args.max_query:
                #     break
    
    test_inputs = []
    for pair in all_test_pairs:
        ((p1, p2), step) = pair
        if random.random() > 0.5:
            input_txt = "Sentence 1: " + inp[p1] + "\nSentence 2: " + inp[p2]
            # for analyzing purpose
            if labels[p1] == labels[p2]:
                output = "Yes"
            else:
                output = "No"
            test_inputs.append({
                "input": input_txt,
                "output": output,
                "options": ['Yes', 'No'],
                "task": args.dataset,
                "sent1_idx": int(p1),
                "sent2_idx": int(p2),
                "num_clusters": int(step)
            })
        else:
            input_txt = "Sentence 1: " + inp[p1] + "\nSentence 2: " + inp[p2]
            # for analyzing purpose
            if labels[p1] == labels[p2]:
                output = "Yes"
            else:
                output = "No"
            test_inputs.append({
                "input": input_txt,
                "output": output,
                "options": ['Yes', 'No'],
                "task": args.dataset,
                "sent1_idx": int(p1),
                "sent2_idx": int(p2),
                "num_clusters": int(step)
            })
    
    with open(save_path, 'w') as f:
        json.dump({
            "test_inputs": test_inputs,
            "clusters": clusters,
            "nodes": nodes,
            "children": children.tolist()
        }, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--embed_method", type=str, default='instructor')
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--feat_path", type=str, required=True)
    # parser.add_argument("--num_data", type=int, default=1024)
    parser.add_argument("--scale", default="small", type=str)
    parser.add_argument("--k", type=int, default=1,
                        help="# of times to sample from a pair of clusters")
    parser.add_argument("--out_dir", default="links", type=str)
    parser.add_argument("--min_clusters", default=2, type=int)
    parser.add_argument("--max_clusters", default=200, type=int)
    parser.add_argument("--seed", type=int, default=100)

    args = parser.parse_args()
    generate(args)
