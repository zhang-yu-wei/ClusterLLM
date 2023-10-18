# we set a window and search for the best range
import os
import json
import argparse
import numpy as np
from sklearn.metrics import fbeta_score

def load_data(args):
    data_path = args.data_path
    return [json.loads(l) for l in open(data_path, 'r')]

def predict(args):
    with open(args.clustering_results, 'r') as f:
        clustering_results = json.load(f)
    # each clustering saves a set of nodes, where each node is a cluster
    # clustering[-1] only has one node
    # clustering[-2] saves two nodes
    # clustering[-3] saves three nodes
    # ...
    # children saves the two nodes that have been merged
    # children[-1] saves two nodes that were merged to create one clusters
    # children[-2] saves two nodes that were merged to create two clusters
    # ...
    # nodes saves the indices that belong to the cluster
    clustering = clustering_results['clusters']
    nodes = clustering_results['nodes']
    children = clustering_results['children']
    with open(args.pred_path, 'r') as f:
        preds = json.load(f)
        # breakpoint()
    # notice that the sampling code of large version has a bug
    # but it only affects the first clustering
    # here is a simple fix
    if args.scale == "large":
        clustering[0] = list(range(0, args.max_clusters))
    
    if "num_clusters" in preds:
        num_clusters_range = preds["num_clusters"]
    else:
        num_clusters_range = list(range(args.min_clusters, args.max_clusters+1))
    
    if isinstance(preds, dict):
        preds = preds['test_inputs']
    # preds = preds[:198]
    # breakpoint()
    
    assign = {}
    for cls_idx in num_clusters_range:
        clst = clustering[-cls_idx]
        cur_assign = {}
        for nidx, node in enumerate(clst):
            # breakpoint()
            # notice that each node save the indices of data that belongs to it
            for idx in nodes[str(node)]:
                cur_assign[idx] = nidx
        assign[cls_idx] = cur_assign
    # breakpoint()
    
    acc = []
    # for cls_idx in assign:
    for cls_idx in num_clusters_range:
        # cur_acc = []
        cur_preds = []
        cur_assign_pair = []
        sample_weights = []
        cur_assign = assign[cls_idx]
        for pred in preds:
            sent1_assign = cur_assign[pred['sent1_idx']]
            sent2_assign = cur_assign[pred['sent2_idx']]
            if 'prediction' in pred and len(pred['prediction']) == 1 and pred['num_clusters']+1 in num_clusters_range:
                # if pred['prediction'][0] == 'Yes' and sent1_assign == sent2_assign:
                #     cur_acc.append(1)
                # elif pred['prediction'][0] == 'No' and sent1_assign != sent2_assign:
                #     cur_acc.append(1)
                # else:
                #     cur_acc.append(0)
                cur_preds.append(1 if pred['prediction'][0] == 'Yes' else 0)                        
                cur_assign_pair.append(1 if sent1_assign == sent2_assign else 0)
                n1, n2 = children[-pred['num_clusters']]
                # sample_weights.append(len(nodes[str(n1)]) * len(nodes[str(n2)]))
                sample_weights.append(np.sqrt(len(nodes[str(n1)]) * len(nodes[str(n2)])))
                # sample_weights.append(len(nodes[str(n1)]) * len(nodes[str(n2)]))
                # breakpoint()
        acc.append(fbeta_score(cur_preds, cur_assign_pair, pos_label=1, beta=0.92))
        # acc.append(fbeta_score(cur_preds, cur_assign_pair, pos_label=1, beta=0.9))
    
    final_k = num_clusters_range[np.argsort(acc)[::-1][0]]
    # final_assign = [assign[final_k][idx] for idx in range(len(assign[final_k]))]
    labels = [l['label'] for l in load_data(args)]
    print(args.dataset)
    print("real k", len(set(labels)))
    print(final_k)
    print(np.argsort(acc)[::-1][:10]+num_clusters_range[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--embed_method", type=str, default=None)
    parser.add_argument("--scale", type=str, default="small")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--clustering_results", type=str, default=None)
    parser.add_argument("--pred_path", type=str, default=None)
    parser.add_argument("--min_clusters", type=int, default=2)
    parser.add_argument("--max_clusters", type=int, default=200)
    args = parser.parse_args()

    predict(args)