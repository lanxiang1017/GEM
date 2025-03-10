from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoTokenizer
# torch.manual_seed(1234)
import torch
# torch.multiprocessing.set_start_method('spawn')
def load_model(model_name, model_args, use_accel=False):
    model_path = model_args.get('model_path_or_name')
    tp = model_args.get('tp', 8)
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(tp))
    model_components = {}
    if use_accel:
        model_components['use_accel'] = True
        model_components['model'] = LLM(model=model_path, tokenizer=model_path, gpu_memory_utilization=0.95, tensor_parallel_size=tp, trust_remote_code=True, disable_custom_all_reduce=True, enforce_eager=True)
        model_components['tokenizer'] = AutoProcessor.from_pretrained(model_path)
        model_components['model_name'] = model_name
    else:
        pass
        # model_components['use_accel'] = False
        # model_components['model'] = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='auto').eval()
        # model_components['tokenizer'] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # model_components['model_name'] = model_name
    return model_components

def infer(prompts, **kwargs):
    model = kwargs.get('model')
    tokenizer = kwargs.get('tokenizer', None)
    use_accel = kwargs.get('use_accel', True)
    sampling_params = SamplingParams(
        temperature=0.5,
        repetition_penalty=1.05,
        max_tokens=5120,
        stop_token_ids=[],
    )

    if use_accel:
        # print(prompts[0])
        if isinstance(prompts[0], str):
            messages = [[{'role': 'user', 'content': prompt}] for prompt in prompts]
        elif isinstance(prompts[0], dict) and 'prompt' in prompts[0] and 'images' in prompts[0]:
            messages = [[{'role': 'user', 'content': [{"type":"image","image":image, "min_pixels": 224 * 224, "max_pixels": 1280 * 28 * 28} for image in prompt["images"]]+[{"type":"text","text":prompt["prompt"]}]}] for prompt in prompts]
        elif isinstance(prompts[0], dict) and 'prompt' in prompts[0] and 'videos' in prompts[0] and isinstance(prompts[0]['videos'], list):
             messages = [[{'role': 'user', 'content': [{"type":"video","video":prompt["videos"],"fps": 1.0}, {"type":"text","text":prompt["prompt"]}]}] for prompt in prompts]
        elif isinstance(prompts[0], dict) and 'conversations' in prompts[0]:
            pass
        else:
            raise ValueError("Invalid prompts format")
        # processed_texts = [tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=True) for message in messages]
        # image_inputs = [process_vision_info(message)[0] for message in messages]
        # llm_inputs = [{"prompt":prompt, "multi_modal_data":{"image":image}} for prompt, image in zip(processed_texts, image_inputs)]
        if "prompt" in prompts[0] and isinstance(prompts[0]["prompt"],str):        
            llm_inputs=[]
            for message in messages:
                prompt=tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=True) 
                image_inputs, video_inputs = process_vision_info(message)

                mm_data = {}
                if image_inputs is not None:
                    mm_data["image"] = image_inputs
                if video_inputs is not None:
                    mm_data["video"] = video_inputs

                llm_input = {
                    "prompt": prompt,
                    "multi_modal_data": mm_data,
                }
                
                llm_inputs.append(llm_input)
                print(llm_input)
            outputs = model.generate(llm_inputs, sampling_params=sampling_params)
            responses = []
            for output in outputs:
                response = output.outputs[0].text
                # print(response)
                responses.append(response)
        
    else:
        pass
        # responses = []
        # for prompt in prompts:
        #     if isinstance(prompt, str):
        #         query = tokenizer.from_list_format([{'text': prompt}])
        #     elif isinstance(prompt, dict) and 'prompt' in prompt and 'images' in prompt:
        #         query = tokenizer.from_list_format([{'image': image} for image in prompt['images']] + [{'text': prompt['prompt']}])
        #     else:
        #         raise ValueError("Invalid prompts format")
        #     response, history = model.chat(tokenizer, query=query, history=None)
        #     responses.append(response)

    return responses

if __name__ == '__main__':
    prompts = [
        '''HI''',
        '''Hello？''',
    ]
    model_args = {
        'model_path_or_name': '/ML-A100/team/mm/zhangge/models/Qwen-VL-Chat',
        'call_type': 'local',
        'tp': 8
    }
    model_components = load_model("Qwen-VL-Chat", model_args, use_accel=False)
    # model_components = {"model": None, "chat_template": get_chat_template_from_config('')}
    responses = infer(prompts, **model_components)
    for response in responses:
        print(response)
