from data_loader import load_data
from models import load_model, infer
import json
import sys
import argparse
from tqdm import tqdm
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from config_wrapper import ConfigWrapper
from tenacity import RetryError


def check_completed(output_file):
    completed = {}
    no_response_id = []
    try:
        with open(output_file, 'r') as file:
            for line in file:
                # print(line)
                data = json.loads(line)
                response_key = config_wrapper.get('response_key')
                error_key = config_wrapper.get('error_key')
                id_key = config_wrapper.get('id_key')
                if response_key in data and (isinstance(data[response_key], str) or (isinstance(data[response_key], dict) and error_key not in data[response_key]) or (error_key in data[response_key] and 'Request failed: 400' in data[response_key][error_key])):
                    completed[config_wrapper.get_id(data)] = data[response_key]
                else:
                    no_response_id.append(config_wrapper.get_id(data))
    except FileNotFoundError:
        pass  # 文件未找到时忽略
    except json.JSONDecodeError:
        pass  # JSON 解码错误时忽略
    return completed, no_response_id

def infer_batch(model_components, model_name, batch):
    results = []
    prompts = [sample[config_wrapper.get('prompt_key')] for sample in batch]
    # try:
    responses = infer(model_name)(prompts, **model_components)
    for sample, response in zip(batch, responses):
        sample[config_wrapper.get('response_key')] = response
        results.append(sample)
    # except RetryError as e:
    #     # 获取最后一次尝试的异常信息
    #     last_attempt = e.last_attempt
    #     if last_attempt:
    #         exception = last_attempt.exception()
    #         if exception:
    #             # print(f"Error processing {prompts}: {str(exception)}", file=sys.stderr)
    #             print(f"Error: {str(exception)}")
    #             for sample in batch:
    #                 sample[config_wrapper.get('response_key')] = {"error": str(exception)}
    #                 results.append(sample)
    # except Exception as e:
    #     # print(f"Error processing {prompts}: {str(e)}", file=sys.stderr)
    #     print(f"Error: {str(e)}")
    #     for sample in batch:
    #         sample[config_wrapper.get('response_key')] = {"error": str(e)}
    #         results.append(sample)
    return results

def main(model_name='gpt4o', splits='test', modes=['dp', 'pyagent'], output_dir='results', infer_limit=None, num_workers=1, batch_size=4, use_accel=False):
    print('-'*100)
    print("[INFO] model_name:", model_name)
    print("[INFO] splits:", splits)
    print("[INFO] modes:", modes)
    print("[INFO] output_dir:", output_dir)
    print("[INFO] Infer Limit:", "No limit" if infer_limit is None else infer_limit)
    print("[INFO] Number of Workers:", num_workers)
    print("[INFO] Batch Size:", batch_size)
    print("[INFO] Use Accel:", use_accel)
    print('-'*100)
    model_components = None
    
    os.makedirs(output_dir, exist_ok=True)
    for split in splits:
        for mode in modes:
            output_file_path = f'{output_dir}/{model_name}_{split}_{mode}.jsonl'
            temp_output_file_path = f'{output_file_path}.tmp'
            
            completed, _ = check_completed(output_file_path)
            temp_completed, _ = check_completed(temp_output_file_path)
            # print(f'Found {len(completed)} completed inferences for {split} {mode} mode.')
            # print(completed)
            merged = {**temp_completed, **completed}
            # print(f'Found {len(merged)} completed inferences for {split} {mode} mode.')
            infer_count = 0

            with open(temp_output_file_path, 'w') as temp_file:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = []
                    batch = []
                    for prompt, sample in tqdm(load_data(split=split, mode=mode), desc=f'Processing {mode}'):
                        # id_key = config_wrapper.get('id_key')
                        sample[config_wrapper.get('prompt_key')] = prompt
                        if config_wrapper.get_id(sample) in merged:
                            
                            sample[config_wrapper.get('response_key')] = merged[config_wrapper.get_id(sample)]
                            json.dump(sample, temp_file, ensure_ascii=False)
                            temp_file.write('\n')
                            temp_file.flush()
                            continue
                        if infer_limit is not None and infer_count >= infer_limit:
                            break
                        if model_components is None:
                            model_components = load_model(model_name, use_accel)
                        batch.append(sample)
                        infer_count += 1
                        if len(batch) == batch_size:
                            futures.append(executor.submit(infer_batch, model_components, model_name, batch.copy()))
                            batch = []
                        if infer_limit is not None and infer_count >= infer_limit:
                            break

                    if batch:
                        futures.append(executor.submit(infer_batch, model_components, model_name, batch))

                    for future in tqdm(as_completed(futures), total=len(futures), desc=f'Writing {mode} results'):
                        results = future.result()
                        print("#####saving......######")
                        for result in results:
                            # print(result)
                            json.dump(result, temp_file, ensure_ascii=False)
                            temp_file.write('\n')
                            temp_file.flush()
            
            # Only rename the temp file to the final output file if the entire process completes successfully
            shutil.move(temp_output_file_path, output_file_path)
            _, no_response_id = check_completed(output_file_path)
            if len(no_response_id) > 0:
                print(f"Failed to get response for {len(no_response_id)} questions in {mode} mode. IDs: {no_response_id}", file=sys.stderr)
        print(f'Inference for {split} completed.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference and save results.')
    parser.add_argument('--model_name', type=str, default='qwen-vl-chat', help='Model name to use')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file to use')
    parser.add_argument('--split', nargs='+', default=['test'], help='Data split to use')
    parser.add_argument('--mode', nargs='+', default=['none'], help='Modes to use for data loading, separated by space')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to write results')
    parser.add_argument('--infer_limit', type=int, help='Limit the number of inferences per run, default is no limit', default=None)
    parser.add_argument('--num_workers', type=int, default=1, help='Number of concurrent workers for inference')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--use_accel', action='store_true', help='Use inference acceleration framework for inference, LLM-->vLLM, VLM-->lmdeploy')
    args = parser.parse_args()
    config_wrapper = ConfigWrapper(args.config)

    main(model_name=args.model_name, splits=args.split, modes=args.mode, output_dir=args.output_dir, infer_limit=args.infer_limit, num_workers=args.num_workers, batch_size=args.batch_size, use_accel=args.use_accel)
