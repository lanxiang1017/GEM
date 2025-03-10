import json
import yaml
import os
import random
random.seed(42)
IMAGE_ROOT="/path/to/PULSE/data/ECGBench/images"

def read_json_or_jsonl(data_path, split='', mapping_key=None):
    base_path = f'{data_path}/{split}'
    if os.path.exists(f'{base_path}.json'):
        file_path = f'{base_path}.json'
    elif os.path.exists(f'{base_path}.jsonl'):
        file_path = f'{base_path}.jsonl'
    else:
        raise FileNotFoundError(f"No JSON or JSONL file found of {base_path}.")
    
    with open(file_path, 'r') as file:
        if file_path.endswith('.json'):
            data = json.load(file)
        elif file_path.endswith('.jsonl'):
            data = [json.loads(line) for line in file]
    
    if mapping_key:
        return {item[mapping_key]: item for item in data if mapping_key in item}
    else:
        return data

# Read the evaluate prompt template from YAML file, we don't need to load the template in default
def read_yaml(config='default'):
    with open(f'config/prompt/{config}.yaml', 'r') as yaml_file:
        return yaml.safe_load(yaml_file)

# Load the data
def load_data(split='ptb-test', mode=None):
    if split in ["code15-test","cpsc-test","csn-test-no-cot","ecgqa-test","g12-test-no-cot","mmmu-ecg","ptb-test","ptb-valid","ptb-test-report"] and mode in ['none']:
        samples = read_json_or_jsonl(f'/path/to/PULSE/data/ECGBench', split)
        for sample in samples:
            prompt = {}
            prompt["prompt"] = sample['conversations'][0]["value"]
            prompt["images"] = [f"{IMAGE_ROOT}/{sample['image']}"]
            prompt['id'] = sample["id"]
            yield prompt, sample
    
    elif split in ["arena"] and mode in ['none']:
        samples = read_json_or_jsonl(f'data/ecg_final_bench', split)
        for sample in samples:
            prompts = {"id":sample["id"],
                       "multi_turn":True,
                       "conversations":[]}
            for turn_id, turn in enumerate(sample['conversations']):
                prompt = {}
                prompt["prompt"] = turn["question"]
                if turn_id==0:
                    prompt["images"] = [f"{IMAGE_ROOT}/{sample['image']}"]
                prompts["conversations"].append(prompt)
            
            yield prompts, sample 

    # for models that refuse to generate reports
    elif split in ["ptb-test-report"] and mode in ['no-refuse']:
        samples = read_json_or_jsonl(f'/path/to/PULSE/data/ECGBench', split)
        template = read_yaml(mode)
        for sample in samples:
            prompt = {}
            prompt["prompt"] = template.format(prompt=sample['conversations'][0]["value"])
            prompt["images"] = [f"{IMAGE_ROOT}/{sample['image']}"]
            prompt['id'] = sample["id"]
            yield prompt, sample
    

if __name__ == '__main__':
    for prompt, sample in load_data('ptb-test'):
        print(prompt)
        break