from openai import OpenAI
import requests
from utils.vl_utils import make_interleave_content

def load_model(model_name="GPT4", base_url="", api_key="", model="gpt-4-turbo-preview"):
    model_components = {}
    model_components['model_name'] = model_name
    model_components['model'] = model
    model_components['base_url'] = base_url
    model_components['api_key'] = api_key
    return model_components

def request(prompt, timeout=120, max_tokens=2000, base_url="", api_key="", model="gpt-4-turbo-preview", model_name=None):
    client = OpenAI(base_url=base_url, api_key=api_key)
    include_system = False
    if model_name == 'baichuan4':
        include_system = True
    if isinstance(prompt, dict) and 'system_prompt' in prompt and 'user_prompt' in prompt:
        system_prompt = prompt['system_prompt']
        user_prompt = prompt['user_prompt']
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    else:
        messages = [{"role": "user", "content": prompt}]
    messages = [{"role": "system", "content": "你是一个有用的助手。"}] * include_system + messages
    # print(messages)
    response = client.chat.completions.create(
        model=model,
        messages = messages,
        stream=False, max_tokens=max_tokens, timeout=timeout)
    # print(response)
    return response

def request_with_images(texts_or_image_paths, timeout=60, max_tokens=2000, base_url="", api_key="", model="gpt-4o", model_name=None):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": make_interleave_content(texts_or_image_paths),
            }  
        ],  
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return response

def request_with_interleave_content(interleave_content, timeout=60, base_url="", api_key="", model="", model_name=None):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    payload = {
        "model": model,
        "messages": interleave_content,
        "max_tokens": 8192
        }

    response = client.chat.completions.create(
        model=model,
        messages=interleave_content,  
        timeout=timeout,
    )
    
    return response["choices"][0]["message"]["content"]

def infer(prompts, **kwargs):
    model = kwargs.get('model')
    base_url = kwargs.get('base_url')
    api_key = kwargs.get('api_key')
    model_name = kwargs.get('model_name', None)
    # try:
    # print("#### here is prompts",prompts)
    # print(prompts[0].keys())
    if isinstance(prompts, list) and "few-shot" not in prompts[0].keys():
        prompts = prompts[0]
    elif "few-shot" in prompts[0].keys():
        
        responses = []
        for data_id, prompt_set in enumerate(prompts):
            print("### ",data_id,"\n")
            prompt_set = prompt_set["conversations"]
            history = None
            # conversations = []
            # print("### prompt_set",prompt_set)
            messages = []
            for idx, data in enumerate(prompt_set):
                # print("=====history\n")
                # print("prompt set data",data)
                user_payload = {"role": "user", "content": None}
                question = data["prompt"]
            
                images = data["images"] if "images" in data.keys() else []
                
                interleave_content = make_interleave_content([question] + images)
                user_payload["content"] = interleave_content
                
                messages.append(user_payload)
                
                if idx != len(prompt_set) - 1:
                    # assistant_payload = {"role": "assistant", "content": {"type": "text", "text": data["response"]}}
                    assistant_payload = {"role": "assistant", "content": data["response"]}
                    messages.append(assistant_payload)
            # print("#### messages",messages)
            response = request_with_interleave_content(messages)
            responses.append(response)
        return responses
    if isinstance(prompts, dict) and 'images' in prompts and "few-shot" not in prompts: 
        prompts, images = prompts['prompt'], prompts['images']
        images = ["<|image|>" + image for image in images]
        response = request_with_images([prompts, *images], base_url=base_url, api_key=api_key, model=model, model_name=model_name).choices[0].message.content

    else:
        response = request(prompts, base_url=base_url, api_key=api_key, model=model, model_name=model_name).choices[0].message.content
    # except Exception as e:
    #     response = {"error": str(e)}
    print(response)
    return [response]

