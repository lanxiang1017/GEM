import json
from openai import OpenAI
import jinja2
from tqdm import tqdm
import os
import argparse

def call_openai_api(prompt, api_key, version):
    client = OpenAI(api_key = api_key)

    completion = client.chat.completions.create(
        model=version,
        store=True,
        messages=[
            {"role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
             }
        ],
    )

    return completion.choices[0].message.content.strip()
 

def construct_prompt(template_raw, val_dic):
    template = jinja2.Template(template_raw, trim_blocks=True, lstrip_blocks=True)
    return template.render(
        generated=val_dic['GEM_generated'],
        groundtruth=val_dic['GPT4o_generated'],
    )

if __name__ == "__main__":

    api_key = ""

    with open("GEM_Evaluation_PTBXL/prompts_evaluation.txt", "r") as f:
        template_raw = f.read()

    ## ! results path
    my_model_version = "gem7b"
    file_path = f"GEM_Evaluation_PTBXL/raw_results/{my_model_version}_ptbxl_results.json"

    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument("-i", "--start", type=int, required=True, help="start index")
    parser.add_argument("-o", "--end", type=int, required=True, help="end index")
    args = parser.parse_args()

    model_version = "gpt-4o-2024-08-06"

    save_dir = f"GEM_Evaluation_PTBXL/gpt_evaluated/{my_model_version}/batch_{args.start}_to_{args.end}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    batch = json_data[args.start:args.end]
    for index, val_dic in tqdm(enumerate(batch)):
        ecg_id = val_dic["id"]

        rst_dic = {}
        rst_dic["id"] = ecg_id

        print(f"processing {index}/{len(batch)} instances...")

        prompt = construct_prompt(template_raw, val_dic)

        result = call_openai_api(prompt, api_key, model_version)

        rst_dic["results"] = result

        file_path = os.path.join(save_dir, f"{ecg_id}.json")

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(rst_dic, file, ensure_ascii=False, indent=4)
    