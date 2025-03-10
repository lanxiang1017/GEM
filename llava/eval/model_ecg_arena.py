import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    # tokenizer.pad_token_id=tokenizer.eos_token_id
    # questions = [json.loads(q) for q in open(args.question_file, "r")]
    
    questions = []
    with open(args.question_file, "r") as f:
        json_data = json.load(f)
        for line in json_data:
            questions.append(line)
            # questions.append({"question_id": line["id"], 
            #                   "image": line["image"], 
            #                   "text": line["conversations"][0]["value"],
            #                   "ans": line["conversations"][1]["value"]})

    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # answers_file = args.answers_file
    # os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    def wrap_first_query(qs):
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        return qs
    print("sampling parameters", args.temperature, args.top_p, args.num_beams, args.max_new_tokens)
    def model_generate(input_ids, image_tensor):
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
    
    ans_file = open(args.answers_file, "w")
    for line in tqdm(questions):
        idx = line["id"]
        print("### ",idx)

        image_file = line["image"]
        # qs = line["text"]
        convs = line["conversations"]
        conv_history = conv_templates[args.conv_mode].copy()
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]
        
        model_conv = []
        for i, conv in enumerate(convs):
            cur_prompt = conv["question"]
            # print(cur_prompt)
            if i==0:
                qs = wrap_first_query(cur_prompt)
                # print("prompt",qs)
                conv_history.append_message(conv_history.roles[0], qs)
                conv_history.append_message(conv_history.roles[1], None)
                prompt = conv_history.get_prompt()
                # print("prompt",prompt)
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                model_response = model_generate(input_ids, image_tensor)
                conv_history.append_message(conv_history.roles[1], model_response)
                model_conv.append({"question": cur_prompt, "answer": model_response})
            else:
                qs = cur_prompt
                # print(conv_history)
                conv_history.append_message(conv_history.roles[0], qs)
                conv_history.append_message(conv_history.roles[1], None)
                prompt = conv_history.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                model_response = model_generate(input_ids, image_tensor)
                conv_history.append_message(conv_history.roles[1], model_response)
                model_conv.append({"question": cur_prompt, "answer": model_response})
        
        print(model_conv)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "golden_convs": convs,
                                   "predict_convs": model_conv,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
        # break
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    eval_model(args)
