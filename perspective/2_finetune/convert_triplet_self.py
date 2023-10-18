import json, os
import h5py
import argparse
import random
from copy import deepcopy
import numpy as np
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=None, type=str)
parser.add_argument("--pred_path", default=None, type=str)
parser.add_argument("--feat_path", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument("--data_path", default=None, type=str)
args = parser.parse_args()

with open(args.pred_path, 'r') as f:
    pred_data = json.load(f)

with open(args.data_path, 'r') as f:
    inp_data = [json.loads(l) for l in f.readlines()]

# the same prompt because it is used for clustering on the same dataset
with open("prompts.json", 'r') as f:
    prompt = json.load(f)[args.dataset]

with h5py.File(args.feat_path, 'r') as f:
    embeds = f['embeds']
    embeds = np.asarray(embeds)

out_data = []
for pd in pred_data:
    # {'query': ['Represent the Wikipedia question for retrieving relevant documents;', 'big little lies season 2 how many episodes'], 'pos': ['Represent the Wikipedia document for retrieval;', 'Big Little Lies (TV series) series garnered several accolades. It received 16 Emmy Award nominations and won eight, including Outstanding Limited Series and acting awards for Kidman, Skarsgård, and Dern. The trio also won Golden Globe Awards in addition to a Golden Globe Award for Best Miniseries or Television Film win for the series. Kidman and Skarsgård also received Screen Actors Guild Awards for their performances. Despite originally being billed as a miniseries, HBO renewed the series for a second season. Production on the second season began in March 2018 and is set to premiere in 2019. All seven episodes are being written by Kelley'], 'neg': ['Represent the Wikipedia document for retrieval;', 'Little People, Big World final minutes of the season two-A finale, "Farm Overload". A crowd had gathered around Jacob, who was lying on the ground near the trebuchet. The first two episodes of season two-B focus on the accident, and how the local media reacted to it. The first season of "Little People, Big World" generated solid ratings for TLC (especially in the important 18–49 demographic), leading to the show\'s renewal for a second season. Critical reviews of the series have been generally positive, citing the show\'s positive portrayal of little people. Conversely, other reviews have claimed that the show has a voyeuristic bend'], 'task_name': 'NQ'}
    choice1_dist = ((embeds[pd['query_idx']] - embeds[pd['choice1_idx']]) ** 2).sum()
    choice2_dist = ((embeds[pd['query_idx']] - embeds[pd['choice2_idx']]) ** 2).sum()
    if choice1_dist < choice2_dist:
        pos = inp_data[pd['choice1_idx']]['input']
        neg = inp_data[pd['choice2_idx']]['input']
    else:
        neg = inp_data[pd['choice1_idx']]['input']
        pos = inp_data[pd['choice2_idx']]['input']
    out_data.append({
        'query': [prompt, inp_data[pd['query_idx']]['input']],
        'pos': [prompt, pos],
        'neg': [prompt, neg],
        'task_name': args.dataset,
        'query_idx': pd['query_idx'],
        'choice1_idx': pd['choice1_idx'],
        'choice2_idx': pd['choice2_idx']
    })

print(out_data[0])
print(out_data[-10])

output_path = os.path.join(args.output_path, args.pred_path.split("/")[-1].replace("-pred.json", "-self-train.json"))
assert not os.path.exists(output_path)
print(output_path)
with open(output_path, 'w') as f:
    json.dump(out_data, f)