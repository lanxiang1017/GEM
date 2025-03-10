import argparse
import torch
import os
import json
import shortuuid
import numpy as np
import math
import torch.multiprocessing as mp
from tqdm import tqdm
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import transformers
import re
from PIL import Image
import wfdb

def split_list(lst, n):
    """Split a list into n roughly equal chunks."""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> torch.Tensor:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i, text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i < len(texts) - 1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i == IMAGE_TOKEN_INDEX for i in _input_id]) == num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = (
                [im_start]
                + [IGNORE_INDEX] * len(tokenizer(role).input_ids)
                + _input_id[len(tokenizer(role).input_ids) + 1: -2]
                + [im_end]
                + nl_tokens
            )
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids

def eval_model_chunk(args, questions_chunk, gpu_id, output_file):
    # Set the GPU for this subprocess.
    torch.cuda.set_device(gpu_id)
    disable_torch_init()
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    all_answers = []
    
    for line in tqdm(questions_chunk, desc=f"GPU {gpu_id}"):
        idx = line["question_id"]
        print("###", idx)
        
        image_file = line["image"]
        ecg_file = line["ecg"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        if "qwen" in model_name.lower():
            conv_0 = line["conversations"][0]
            conv_0['value'] = '<image>\n' + conv_0['value']
            input_ids = preprocess_qwen([conv_0, {'from': 'gpt', 'value': None}], tokenizer, has_image=True).cuda()
        else:
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        image_path = os.path.join(args.image_folder, image_file)
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]
        
        ecg_folder = args.ecg_folder
        ecg_path = os.path.join(ecg_folder, ecg_file)
        ecg = wfdb.rdsamp(ecg_path)[0]
        ecg[np.isnan(ecg)] = 0
        ecg[np.isinf(ecg)] = 0
        ecg = torch.Tensor(np.transpose(ecg, (1, 0)).astype(np.float32))
        c, length = ecg.shape
        seq_length = 5000
        if length < seq_length:
            new_ecg = torch.zeros((c, seq_length))
            new_ecg[:, 0:length] = ecg
            ecg = new_ecg
        elif length > seq_length:
            ecg = ecg[:, 0:seq_length]
        ecg = ecg.half()
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                ecgs=ecg,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True
            )
        
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(outputs)
        
        ans_id = shortuuid.uuid()
        new_answer = {
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {}
        }
        all_answers.append(new_answer)
    
    # Write the results for this chunk to a temporary file.
    with open(output_file, "w") as ans_file:
        for answer in all_answers:
            ans_file.write(json.dumps(answer) + "\n")

def merge_results(num_processes, merged_output_file):
    all_answers = []
    for gpu_id in range(num_processes):
        temp_file = f"answers_gpu{gpu_id}.jsonl"
        if os.path.exists(temp_file):
            with open(temp_file, "r") as f:
                for line in f:
                    all_answers.append(json.loads(line))
    with open(merged_output_file, "w") as fout:
        for answer in all_answers:
            fout.write(json.dumps(answer) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--ecg-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1") # qwen_2 for QWen model
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--ecg_tower", type=str, default="")
    parser.add_argument("--open_clip_config", type=str, default="")
    args = parser.parse_args()


    questions = []
    with open(args.question_file, "r") as f:
        json_data = json.load(f)
        for line in json_data:
            # print(line)
            questions.append({"question_id": line["id"], 
                                "image": line["image"], 
                                "text": line["conversations"][0]["value"].replace("<image>\n",""),
                                "ans": line["conversations"][1]["value"],
                                "ecg": line["ecg"],
                                "conversations": line["conversations"],
                                },
                                )
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("No GPUs found.")
    
    # divide questions into chunks, one for each GPU.
    chunks = split_list(questions, num_gpus)
    
    processes = []
    for gpu_id in range(num_gpus):
        temp_file = f"answers_gpu{gpu_id}.jsonl"
        p = mp.Process(target=eval_model_chunk, args=(args, chunks[gpu_id], gpu_id, temp_file))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    # merge temporary answer files into the final answers file.
    merge_results(num_gpus, args.answers_file)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
