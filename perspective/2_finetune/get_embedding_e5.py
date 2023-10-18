import os
import torch
import json
import h5py
import argparse
import numpy as np
import tqdm
from functools import partial
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerFast, BatchEncoding
from transformers.modeling_outputs import BaseModelOutput
from typing import List, Dict

from e5_utils import logger, pool, move_to_cuda
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment

def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc

def clustering_score(y_true, y_pred):
    return {'ACC': clustering_accuracy_score(y_true, y_pred)*100,
            'ARI': adjusted_rand_score(y_true, y_pred)*100,
            'NMI': normalized_mutual_info_score(y_true, y_pred)*100}


parser = argparse.ArgumentParser(description='evaluation for MTEB benchmark except its Retrieval category')
parser.add_argument('--input_path', default=None, type=str)
parser.add_argument('--output_path', default=None, type=str)
parser.add_argument('--model-name-or-path', default='tmp-outputs/',
                    type=str, metavar='N', help='which model to use')
parser.add_argument("--checkpoint", default=None, type=str)
parser.add_argument('--l2-normalize', action='store_true', help='whether to l2 normalize embeddings')
parser.add_argument('--pool-type', default='avg', help='pool type')
parser.add_argument('--prompt', default='query: ', help='prompt')
parser.add_argument("--measure", action="store_true",
                    help="if measure clustering performance")
parser.add_argument("--scale", default="small", type=str)

args = parser.parse_args()
logger.info('Args: {}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
assert args.prompt in ['', 'query: ', 'passage: ']

def _transform_func(tokenizer: PreTrainedTokenizerFast,
                    examples: Dict[str, List]) -> BatchEncoding:
    if args.prompt:
        examples['input_texts'] = [args.prompt + t for t in examples['input_texts']]
    batch_dict = tokenizer(examples['input_texts'],
                           max_length=512,
                           padding=True,
                           truncation=True)

    return batch_dict

class DenseEncoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path)
        if args.checkpoint is not None:
            print(f"Loading from {args.checkpoint} ...")
            # state_dict = torch.load(os.path.join(args.checkpoint, 'pytorch_model.bin'))
            # self.encoder.load_state_dict(state_dict)
            self.encoder = AutoModel.from_pretrained(args.checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.gpu_count = torch.cuda.device_count()

        self.encoder.eval()
        self.encoder.cuda()

        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)

    @torch.no_grad()
    def encode(self, sentences, **kwargs) -> np.ndarray:
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """

        dataset: Dataset = Dataset.from_dict({'input_texts': sentences})
        dataset.set_transform(partial(_transform_func, self.tokenizer))

        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        data_loader = DataLoader(
            dataset,
            batch_size=128 * self.gpu_count,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            collate_fn=data_collator,
            pin_memory=True)

        encoded_embeds = []
        for batch_dict in tqdm.tqdm(data_loader, desc='encoding', mininterval=10, disable=len(sentences) < 128):
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], args.pool_type)
                if args.l2_normalize:
                    embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)


def _convert_label_to_ids(labels):
    unique_labels = list(set(labels))
    n_clusters = len(unique_labels)
    label_map = {l: i for i, l in enumerate(unique_labels)}
    label_ids = [label_map[l] for l in labels]
    return np.asarray(label_ids), n_clusters


def eval_embeds(embeds, labels, args):
    if labels is not None:
        label_ids, n_clusters = _convert_label_to_ids(labels)
        
        all_measures = {'ACC': [], 'NMI': [], 'ARI': []}
        for seed in [100, 13, 21, 36, 42]:
            if args.scale == "small":
                logger.info(f"Fitting K-Means model (seed: {seed})...")
                preds = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(embeds)
            elif args.scale == "large":
                logger.info(f"Fitting MiniBatch K-Means model (seed: {seed})...")
                preds = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed).fit_predict(embeds)
            preds = np.asarray(preds)
            measures = clustering_score(label_ids, preds)
            for k in measures:
                all_measures[k].append(measures[k])

        for k in ['ACC', 'NMI', 'ARI']:
            # print(k)
            mean = np.mean(all_measures[k])
            # print("Mean: ", round(mean, 2))
            std = np.std(all_measures[k])
            # print("Std: ", round(std, 2))

            all_measures[f'{k}_mean'] = mean
            all_measures[f'{k}_std'] = std
    else:
        all_measures = {}

    return all_measures


def main():
    model = DenseEncoder()
    assert args.input_path.endswith("jsonl"), "input file must be jsonl"
    assert args.output_path.endswith("hdf5"), "output file must be hdf5"
    assert "e5" in args.output_path

    with open(args.input_path, 'r') as f:
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

    if not os.path.exists(args.output_path):
        # breakpoint()
        
        embeds = model.encode(texts)

        # evaluator = ClusteringEvaluator(sentences=texts, labels=labels, args=args)
        # measures, embeds = evaluator(model)
        measures = eval_embeds(embeds, labels, args)

        with h5py.File(args.output_path, 'w') as f:
            dset = f.create_dataset("embeds", data=embeds)
    
    else:
        with h5py.File(args.output_path, 'r') as f:
            embeds = np.asarray(f['embeds'])
        # evaluator = ClusteringEvaluator(sentences=texts, labels=labels, args=args)
        # measures = evaluator.eval_only(embeds)
        measures = eval_embeds(embeds, labels, args)
    
    if measures is not None and args.measure:
        with open(args.output_path.replace(".hdf5", "_measures.json"), 'w') as f:
            json.dump(measures, f)

    print(measures)

    print("--DONE--")

if __name__ == '__main__':
    main()