import os
import json
import argparse
from tqdm import tqdm

def validate_and_repair_jsonl(file_path, repair=False, overwrite=False):
    """
    Validate and optionally repair a JSONL file.
    """
    valid_lines = []
    invalid_lines = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i, line in tqdm(enumerate(lines), total=len(lines), desc="Processing"):
            try:
                valid_line = json.loads(line.strip())
                valid_lines.append(valid_line)
            except json.JSONDecodeError:
                invalid_lines.append(i + 1)
    
    if invalid_lines:
        print('-' * 150)
        print(f"File {file_path} contains {len(invalid_lines)} invalid lines, at lines {invalid_lines}.")
        print('-' * 150)
        
        if repair:
            save_path = file_path if overwrite else file_path.replace('.jsonl', '_repaired.jsonl')
            with open(save_path, 'w', encoding='utf-8') as file:
                file.writelines(valid_lines)
            print(f"Repaired file saved as {save_path}")
    else:
        # print("All lines are valid JSON.")
        for valid_line in valid_lines:
            if isinstance(valid_line['response'], dict) and 'error' in valid_line['response']:
                print(f"Error in line {valid_line['id']}: {valid_line['response']['error']}")

def main(folder_path, repair=False, overwrite=False):
    for file_path in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_path)
        validate_and_repair_jsonl(file_path, repair, overwrite)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate and repair JSONL files.')
    parser.add_argument('folder_path', type=str, help='Path to the JSONL file.')
    parser.add_argument('--repair', action='store_true', help='Repair the file by removing invalid lines.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite the original file with the repaired file.')

    args = parser.parse_args()
    main(args.folder_path, args.repair, args.overwrite)
