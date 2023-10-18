import json, argparse, os
import random

dataset2lp = {
    "banking77": "intent",
    "few_rel_nat": "relation type",
    "massive_scenario": "scenario",
    "massive_intent": "intent",
    "mtop_domain": "domain",
    "mtop_intent": "intent",
    "clinc": "intent",
    "clinc_domain": "domain"
}

def prepare_prompt(pos_pairs, neg_pairs, label_property):
    prompt_pos = f"[Example<IDX>]\nSentence 1: <SENT1>\nSentence 2: <SENT2>\nYes. Because both {label_property}s are <LABEL>.\n\n"
    prompt_neg = f"[Example<IDX>]\nSentence 1: <SENT1>\nSentence 2: <SENT2>\nNo. Because Sentence 1 has {label_property} <LABEL1> and Sentence 2 has {label_property} <LABEL2>.\n\n"
    inst = f"Determine whether the {label_property}s of two banking customer utterances below belong to the same {label_property} category using above examples.\n\n"

    final_prepared = ""
    for idx, pair in enumerate(pos_pairs):
        prepared = prompt_pos
        prepared = prepared.replace("<IDX>", str(idx + 1))
        prepared = prepared.replace("<SENT1>", pair['sent1'])
        prepared = prepared.replace("<SENT2>", pair['sent2'])
        prepared = prepared.replace("<LABEL>", pair['label'])
        final_prepared += prepared
    for idx, pair in enumerate(neg_pairs):
        prepared = prompt_neg
        prepared = prepared.replace("<IDX>", str(idx + len(pos_pairs) + 1))
        prepared = prepared.replace("<SENT1>", pair['sent1'])
        prepared = prepared.replace("<SENT2>", pair['sent2'])
        prepared = prepared.replace("<LABEL1>", pair['label1'])
        prepared = prepared.replace("<LABEL2>", pair['label2'])
        final_prepared += prepared
    final_prepared += inst
    return final_prepared

def main(args):
    random.seed(args.seed)

    if os.path.exists(args.prompt_path):
        with open(args.prompt_path, 'r') as f:
            all_prompts = json.load(f)
    else:
        all_prompts = {}
    
    with open(args.sampled_pair_path, 'r') as f:
        all_pairs = json.load(f)['test_inputs']
    
    with open(args.data_path, 'r') as f:
        data = [json.loads(l) for l in f.readlines()]
    
    # suppose we sample limited amount of pairs and then annotate all of them
    sampled_pairs = random.sample(all_pairs, args.num_sampled)
    pairs_pos = [p for p in sampled_pairs if p['output'] == 'Yes']
    pairs_neg = [p for p in sampled_pairs if p['output'] == 'No']
    # choose coarse-grained for positive
    pairs_pos = sorted(pairs_pos, key=lambda x: x['num_clusters'])[:args.num_for_prompt]
    pairs_neg = sorted(pairs_neg, key=lambda x: x['num_clusters'], reverse=True)[:args.num_for_prompt]
    for pair in pairs_pos:
        assert data[pair['sent1_idx']]['label'] == data[pair['sent2_idx']]['label']
        pair['label'] = data[pair['sent1_idx']]['label']
        pair['sent1'] = data[pair['sent1_idx']]['input']
        pair['sent2'] = data[pair['sent2_idx']]['input']
    for pair in pairs_neg:
        assert data[pair['sent1_idx']]['label'] != data[pair['sent2_idx']]['label']
        pair['label1'] = data[pair['sent1_idx']]['label']
        pair['label2'] = data[pair['sent2_idx']]['label']
        pair['sent1'] = data[pair['sent1_idx']]['input']
        pair['sent2'] = data[pair['sent2_idx']]['input']
    
    final_prompt = prepare_prompt(pairs_pos, pairs_neg, dataset2lp[args.dataset])
    all_prompts[args.dataset] = final_prompt

    with open(args.prompt_path, 'w', encoding='utf-8') as f:
        json.dump(all_prompts, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ----- path -----
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--sampled_pair_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    # ----- other -----
    parser.add_argument("--dataset", type=str, default="banking77")
    parser.add_argument("--num_sampled", type=int, default=16)
    parser.add_argument("--num_for_prompt", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(args)