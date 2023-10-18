import json
import h5py
import numpy as np

dataset = "DATASET"
scale = "SCALE"

triplet_pred_path = "PATH_TO_TRIPLET_PRED"
embeds_path = f"../../datasets/{dataset}/{scale}_embeds.hdf5"

with open(triplet_pred_path, 'r') as f:
    data = json.load(f)
with h5py.File(embeds_path, 'r') as f:
    embeds = np.asarray(f['embeds'])

llm_acc = []
embed_acc = []
consensus = []
for datum in data:
    answer = datum['output']
    if answer in [' 1', ' 2']:
        query_embed = embeds[datum['query_idx']]
        choice1_embed = embeds[datum['choice1_idx']]
        choice2_embed = embeds[datum['choice2_idx']]
        d_qc1 = ((query_embed - choice1_embed) ** 2).sum()
        d_qc2 = ((query_embed - choice2_embed) ** 2).sum()
        pred_embed = ' 1' if d_qc1 < d_qc2 else ' 2'
        pred_llm = None
        if datum['prediction']:
            pred_llm = datum['prediction'][0]
        if pred_embed == answer:
            embed_acc.append(1)
        else:
            embed_acc.append(0)
        if pred_llm == answer:
            llm_acc.append(1)
        else:
            llm_acc.append(0)
        if pred_llm == pred_embed:
            consensus.append(1)
        else:
            consensus.append(0)

print("LLM Acc: ", np.mean(llm_acc))
print("Embed Acc: ", np.mean(embed_acc))
print("Consensus: ", np.mean(consensus))