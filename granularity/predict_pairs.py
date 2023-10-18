# the model takes in a pair of  datapoints
import os
import json
import argparse
import openai
from tqdm import tqdm
from tools import delayed_completion

def prepare_data(prompt, datum):
    postfix = "\n\nPlease respond with 'Yes' or 'No' without explanation."
    input_txt = datum["input"]
    return prompt + input_txt + postfix

def post_process(completion):
    content = completion['choices'][0]['message']['content'].strip()
    result = []
    if 'Yes' in content and 'No' not in content:
        result.append('Yes')
    elif 'No' in content and 'Yes' not in content:
        result.append('No')
    return content, result

def predict(args):
    openai.organization = args.openai_org
    openai.api_key = os.getenv("OPENAI_API_KEY")

    prompt_file_name = args.prompt_file.split("/")[-1].split(".")[0]

    pred_path = args.data_path.split("/")[-1].replace(".json", f"-{args.model_name}-{prompt_file_name}.json")
    pred_path = os.path.join("predicted_pair_results", pred_path)
    print("Save in: ", pred_path)
    num_clusters = None
    if os.path.exists(pred_path):
        with open(pred_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                if 'num_clusters' in data:
                    num_clusters = data['num_clusters']
                data = data['test_inputs']
    else:
        with open(args.data_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                if 'num_clusters' in data:
                    num_clusters = data['num_clusters']
                data = data['test_inputs']
    
    if args.previous_path is not None:
        with open(args.previous_path, 'r') as f:
            prev_data = json.load(f)
            if isinstance(prev_data, dict):
                prev_data = prev_data['test_inputs']
        prev_inputs = {d['input']: d for d in prev_data}
    else:
        prev_inputs = {}
    
    with open(args.prompt_file, 'r') as f:
        prompts = json.load(f)
        task_prompt = prompts[args.dataset]
    
    for d in data:
        if 'prepared' not in d:
            d['prepared'] = prepare_data(task_prompt, d)
    
    for idx, datum in tqdm(enumerate(data), total=len(data)):
        if idx == 0:
            print(datum['prepared'])
        # breakpoint()
        if 'prediction' in datum:
            continue
        if datum['input'] in prev_inputs:
            data[idx]['content'] = prev_inputs[datum['input']]['content']
            data[idx]['prediction'] = prev_inputs[datum['input']]['prediction']
            continue
        messages = [
            {"role": "user", "content": datum['prepared']}
        ]
        
        completion, error = delayed_completion(delay_in_seconds=args.delay, max_trials=args.max_trials, model=args.model_name, messages=messages, max_tokens=10, temperature=args.temperature)
        if completion is None:
            print(f"Saving data after {idx + 1} inference.")
            with open(pred_path, 'w') as f:
                if num_clusters is not None:
                    data = {"num_clusters": num_clusters, "test_inputs": data}
                json.dump(data, f)
            print(error)
            breakpoint()
        else:
            content, results = post_process(completion)
            data[idx]['content'] = content
            data[idx]['prediction'] =  results
            # breakpoint()
        
        if idx % args.save_every == 0 and idx > 0:
            print(f"Saving data after {idx + 1} inference.")
            with open(pred_path, "w") as f:
                if num_clusters is not None:
                    data = {"num_clusters": num_clusters, "test_inputs": data}
                json.dump(data, f)
            if isinstance(data, dict):
                data = data['test_inputs']
        
        # if idx > 10:
        #     break
    
    with open(pred_path, "w") as f:
        if num_clusters is not None:
            data = {"num_clusters": num_clusters, "test_inputs": data}
        json.dump(data, f)
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--data_path", default=None, type=str)
    parser.add_argument("--openai_org", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt-4")
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--max_trials", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--num_responses", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--previous_path", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, required=True)
    args = parser.parse_args()

    predict(args)
    