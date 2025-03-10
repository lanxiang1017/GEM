import json
from statistics import mean
import os
import requests
import json 
from openai import AzureOpenAI, OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, wait_random_exponential, stop_after_attempt
import concurrent.futures
import json
from tqdm import tqdm
import numpy as np
from promtps import arena_eval_prompt


os.environ["AZURE_OPENAI_API_KEY"] = "api key"
os.environ["AZURE_OPENAI_ENDPOINT"] = "endpoint"

def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def process(datum output_dir, eval_model_name, client):
    
    if "predict_convs" in datum:
        a1, a2 = datum['predict_convs'][0]['answer'], datum['predict_convs'][1]['answer']
    elif "response" in datum:
        a1, a2 = datum['response'][0]['response'], datum['response'][1]['response']
    
    initial_prompt = arena_eval_prompt
    if "question_id" in datum:
        question_no = datum['question_id'].split('-')[-1]
    elif "id" in datum:
        question_no = datum['id'].split('-')[-1]
    
    if "golden_convs" in datum:
        q1, gt_a1 = datum['golden_convs'][0]['question'], datum['golden_convs'][0]['answer']
        q2, gt_a2 = datum['golden_convs'][1]['question'], datum['golden_convs'][1]['answer']
    elif "conversations" in datum:
        q1, gt_a1 = datum['conversations'][0]['question'], datum['conversations'][0]['answer']
        q2, gt_a2 = datum['conversations'][1]['question'], datum['conversations'][1]['answer']
    
    gt_prompt = f"**Question:**\n{q1}\n\n**Ground Truth Answer:** {gt_a1}\n\n**Question:**\n{q2}\n\n**Ground Truth Answer:** {gt_a2}"
    model_prompt = f"**Question:**\n{q1}\n\n**Model's Response:** {a1}\n\n**Question:**\n{q2}\n\n**Model's Response:** {a2}"

    prompt = f"{initial_prompt} \n [The Start of Ground Truth Answer]\n {gt_prompt}\n [The End of Ground Truth Answer]\n\n [The Start of Model's Response]\n {model_prompt}\n [The End of Model's Response]"

    response = client.chat.completions.create(
        model=eval_model_name,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
   
    # Save the JSON response directly to a .json file
    with open(f'{output_dir}/{question_no}.json', 'w') as f:
        f.write(response.choices[0].message.content)
        


def run_pairwise_comparison(arena_response_file, output_dir, eval_model_name, client):
    
    arena_response = load_jsonl(arena_response_file)
    existing_qids = [file.split('.')[0] for file in os.listdir(output_dir)]
    if "question_id" in arena_response[0]:
        filtered_arena_response = [datum for datum in arena_response if datum['question_id'].split('-')[-1] not in existing_qids]
    elif "id" in arena_response[0]:
        filtered_arena_response = [datum for datum in arena_response if datum['id'].split('-')[-1] not in existing_qids]
    print(f"Number of evaluation samples: {len(filtered_arena_response)}")
    print(f"eval_model_name: {eval_model_name}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process, datum, output_dir, eval_model_name, client) for datum in filtered_arena_response]
        # Use tqdm to show progress
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()  # Wait for the result to handle any exceptions that might occur


def compute_score(output_dir):
    # Initialize lists to store scores for each category
    accuracy_scores = []
    completeness_scores = []
    instruction_adherence_scores = []

    # Iterate over each file in the directory
    for filename in os.listdir(output_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(output_dir, filename)
            print(filepath)
            with open(filepath, 'r') as file:
                data = json.load(file)
                
                accuracy_scores.append(data['Accuracy']['Score'] * 10)
                completeness_scores.append(data['Completeness']['Score'] * 10)
                instruction_adherence_scores.append(data['Instruction Adherence']['Score'] * 10)

    # Calculate average scores for each category
    avg_accuracy = mean(accuracy_scores)
    avg_completeness = mean(completeness_scores)
    avg_instruction_adherence = mean(instruction_adherence_scores)

    # Calculate overall average score across all categories
    total_scores = accuracy_scores + completeness_scores + instruction_adherence_scores
    overall_avg_score = mean(total_scores)

    # Print the results
    print(f"Average Accuracy Score: {avg_accuracy}")
    print(f"Average Completeness Score: {avg_completeness}")
    print(f"Average Instruction Adherence Score: {avg_instruction_adherence}")
    print(f"Overall Average Score: {overall_avg_score}")
# bash bench_eval_main_arena.sh -s 25745 -e 25745 -i 2315 -m llava-v1.6-vicuna-7b-ecg-chat-0925-v4_3 -d arena -c 0 -t 0 -f false
# main function
def main():
    model_name = "pulse"
    test_model_name = "step-final"
    arena_response_file = f'/path/to/eval_outputs/{model_name}/arena/{test_model_name}.jsonl'
    
    # arena score save directory
    output_dir = f'/path/to/arena_scores/arena-{model_name}-{test_model_name}'
    
    os.makedirs(output_dir, exist_ok=True)

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2024-08-01-preview",
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    eval_model_name='deploy name' # gpt-4o
    
    print(f"Pairwise Comparison: {model_name}-{test_model_name}")
    run_pairwise_comparison(arena_response_file, output_dir, eval_model_name, client)

    print(f"ECG Arena Score: {output_dir}")
    compute_score(output_dir)

if __name__ == '__main__':
    main()