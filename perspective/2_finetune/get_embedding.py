import os
import sys
import json
import h5py
import torch
import logging
import argparse
import numpy as np
from InstructorEmbedding import INSTRUCTOR
from clustering_utils.evaluator import ClusteringEvaluator

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default=None,type=str)
parser.add_argument('--task_name', default=None, type=str)
parser.add_argument('--data_path', default=None, type=str)
parser.add_argument('--cache_dir', default=None,type=str)
parser.add_argument('--result_file', default=None,type=str)
parser.add_argument('--prompt', default=None,type=str)
parser.add_argument('--batch_size', default=-1,type=int)
parser.add_argument("--checkpoint", default=None, type=str)
parser.add_argument("--scale", default="small", type=str)
parser.add_argument("--measure", action="store_true",
                    help="if measure clustering performance")
parser.add_argument("--overwrite", action="store_true",
                    help="if overwrite the embedding")
args = parser.parse_args()

with open(args.data_path, 'r') as f:
    data = [json.loads(l) for l in f]

texts, labels = [], []
for datum in data:
    texts.append(datum['input'])
    if args.measure and 'label' in datum:
        labels.append(datum['label'])
    elif args.measure and 'label' not in datum:
        raise ValueError("Label not provided!")
    else:
        labels = None

if os.path.exists(args.result_file) and not args.overwrite:
    
    with h5py.File(args.result_file, 'r') as f:
        embeds = np.asarray(f['embeds'])
    evaluator = ClusteringEvaluator(sentences=texts, labels=labels, args=args)
    measures = evaluator.eval_only(embeds)

else:

    model = INSTRUCTOR(args.model_name,cache_folder=args.cache_dir)
    if args.checkpoint is not None:
        print(f"Loading from {args.checkpoint} ...")
        state_dict = torch.load(os.path.join(args.checkpoint, 'pytorch_model.bin'))
        model.load_state_dict(state_dict)

    if args.prompt is None:
        args.prompt = args.model_name
    if not args.prompt in ['hkunlp/instructor-xl','hkunlp/instructor-base']:
        args.prompt = 'hkunlp/instructor-large'

    evaluator = ClusteringEvaluator(sentences=texts, labels=labels, args=args)
    measures, embeds = evaluator(model)

    with h5py.File(args.result_file, 'w') as f:
        dset = f.create_dataset("embeds", data=embeds)
    # lset = f.create_dataset("labels", data=labels)
    

if measures is not None and args.measure:
    with open(args.result_file.replace(".hdf5", "_measures.json"), 'w') as f:
        json.dump(measures, f)

print(measures)

print("--DONE--")