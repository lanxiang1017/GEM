import json
import argparse
from tqdm import tqdm

def load_jsonl(file_path):
    """
    Load JSONL file and return a list of dictionaries.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def build_response_map(file_data):
    """
    Build a map from queries to responses.
    """
    response_map = {}
    for item in file_data:
        response_map[item['query']] = item['responses']
    return response_map

def process_files(file1_data, response_map):
    """
    Process file1 data with reference to response map from file2.
    """
    for item in tqdm(file1_data, desc="Processing"):
        query = item['query']
        if query in response_map:
            responses2 = response_map[query]
            responses1 = item['responses']
            
            # Update missing keys in responses1 with those from responses2
            for key, value in responses2.items():
                if key not in responses1:
                    responses1[key] = value
            
            # Replace 'claude-3-5-sonnet-20240620' field if it is a dict in responses1
            if isinstance(responses1.get('claude-3-5-sonnet-20240620'), dict):
                responses1['claude-3-5-sonnet-20240620'] = responses2.get('claude-3-5-sonnet-20240620', responses1['claude-3-5-sonnet-20240620'])

    return file1_data

def save_jsonl(file_path, data):
    """
    Save a list of dictionaries to a JSONL file.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

def main(file1_path, file2_path, output_path, overwrite=False):
    file1_data = load_jsonl(file1_path)
    file2_data = load_jsonl(file2_path)
    
    response_map = build_response_map(file2_data)
    updated_data = process_files(file1_data, response_map)
    
    save_path = file1_path if overwrite else output_path
    save_jsonl(save_path, updated_data)
    print(f"Processed data saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process two JSONL files as described.')
    parser.add_argument('file1_path', type=str, help='Path to the first JSONL file.')
    parser.add_argument('file2_path', type=str, help='Path to the second JSONL file.')
    parser.add_argument('--output', type=str, default='output.jsonl', help='Path to save the processed file.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite the original file1 with the processed data.')

    args = parser.parse_args()
    main(args.file1_path, args.file2_path, args.output, args.overwrite)
