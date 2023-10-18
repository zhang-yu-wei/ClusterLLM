import os
import json
import random
import h5py
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

def load_data(args):
    # data_path = f"../../{args.dataset}/{args.scale}.jsonl"
    return [json.loads(l) for l in open(args.data_path, 'r')]

def load_feat(args):
    feat_path = args.feat_path
    with h5py.File(feat_path, 'r') as f:
        X = f['embeds']
        X = np.asarray(X)
    return X
    
def entropy(vals):
    vals = np.asarray(vals)
    vals /= vals.sum()
    return - (vals * np.log(vals)).sum()

def generate(args):
    os.makedirs(args.out_dir, exist_ok=True)
    save_path = f"{args.out_dir}/{args.dataset}_embed={args.embed_method}_s={args.scale}_m={args.max_query}_d={round(args.max_distance, 1)}{'_f=' + str(round(args.filter_first_prop, 2)) if args.filter_first_prop != 0 else ''}{'_l=' + str(round(args.large_ent_prop, 2)) if args.large_ent_prop != 0.2 else ''}{'_p=' + str(round(args.close_cluster_prop, 3)) if args.close_cluster_prop != 0.02 else ''}{'_sf' if args.shuffle_inds else ''}_choice_seed={args.seed}.json"
    print(save_path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    data = load_data(args)
    inp = [d['input'] for d in data]
    # for analyzing purpose only
    labels = [d['label'] for d in data]
    X = load_feat(args)

    X = StandardScaler().fit_transform(X)

    if args.scale == "small":
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=args.max_distance).fit(X)
    elif args.scale == "large":
        clustering = MiniBatchKMeans(n_clusters=100, random_state=args.seed).fit(X)
    preds = clustering.labels_
    n_clusters = len(set(preds))
    print("Estimated number of clusters: %d" % n_clusters)

    cluster_centers = []
    class_member_inds = {}
    for i in range(n_clusters):
        class_member_mask = preds == i
        cluster_centers.append(X[class_member_mask].mean(0))
        class_member_inds[i] = np.where(class_member_mask)[0]
    cluster_centers = np.stack(cluster_centers)

    # closest clusters
    num_closest = max(2, round(n_clusters * args.close_cluster_prop))
    options = []
    entropies = []
    for idx in range(len(X)):
        dist = ((X[idx] - cluster_centers) ** 2).sum(-1)
        prob = (1 + dist) ** (-1)
        prob /= prob.sum()
        # select most probable clusters
        sorted_prob = np.argsort(prob)[::-1][:num_closest]
        options.append(sorted_prob) # most probable cluster index
        entropies.append(entropy(prob[sorted_prob])) # entropy with most probable clusters
    if args.filter_first_prop > 0:
        sorted_ent = np.argsort(entropies)[::-1][int(len(X) * args.filter_first_prop):int(len(X) * args.large_ent_prop)]
    else:
        sorted_ent = np.argsort(entropies)[::-1][:int(len(X) * args.large_ent_prop)]
    if args.shuffle_inds:
        np.random.shuffle(sorted_ent)
    
    triplets = []
    while len(triplets) < args.max_query:
        for idx in sorted_ent:
            cur_options = options[idx].tolist()
            # sample 2 from most probable clusters
            cluster1, cluster2 = random.sample(cur_options, 2)
            choice1 = random.choice(class_member_inds[cluster1])
            choice2 = random.choice(class_member_inds[cluster2])
            if (idx, choice1, choice2) not in triplets \
                and choice1 != idx and choice2 != idx:
                triplets.append((idx, choice1, choice2))
                if len(triplets) >= args.max_query:
                    break
    
    # !warning: some of the codes below might be unnecessary
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

    print("Total number: ", len(result))
    with open(save_path, 'w') as f:
        json.dump(result, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--feat_path", type=str, required=True)
    parser.add_argument("--embed_method", type=str, default='instructor')
    parser.add_argument("--scale", type=str, default="small")
    parser.add_argument("--max_query", type=int, default=256)
    parser.add_argument("--large_ent_prop", type=float, default=0.20)
    parser.add_argument("--filter_first_prop", type=float, default=0.)
    parser.add_argument("--close_cluster_prop", type=float, default=0.02)
    parser.add_argument("--max_distance", type=float, default=67)
    parser.add_argument("--shuffle_inds", action="store_true")
    parser.add_argument("--out_dir", default="links", type=str)
    parser.add_argument("--seed", type=int, default=100)

    args = parser.parse_args()
    generate(args)