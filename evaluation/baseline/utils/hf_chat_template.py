import os
import json
import warnings

def _compile_jinja_template(chat_template):
    import jinja2
    from jinja2.exceptions import TemplateError
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise TemplateError(message)

    jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    jinja_env.globals["raise_exception"] = raise_exception
    return jinja_env.from_string(chat_template)
    
def default_chat_template():
    """
    This template formats inputs in the standard ChatML format. See
    https://github.com/openai/openai-python/blob/main/chatml.md
    """
    return (
        "{% for message in messages %}"
        "{{'' + message['role'] + '\n' + message['content'] + '' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ 'assistant\n' }}"
        "{% endif %}"
    )

def get_chat_template_from_config(model_path):
    config_path = os.path.join(model_path, "tokenizer_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get("chat_template", default_chat_template())
        except Exception as e:
            warnings.warn(f"Error reading tokenizer_config.json: {e}")
    else:
        warnings.warn(f"tokenizer_config.json not found in {model_path}. Using default chat template.")
    return default_chat_template()

def render_chat_template(messages, chat_template, add_generation_prompt=True):
    compiled_template = _compile_jinja_template(chat_template)

    if isinstance(messages, (list, tuple)) and (
        isinstance(messages[0], (list, tuple)) or hasattr(messages[0], "messages")
    ):
        is_batched = True
    else:
        messages = [messages]
        is_batched = False

    rendered = []
    for chat in messages:
        if hasattr(chat, "messages"):
            # Indicates it's a Conversation object
            chat = chat.messages
        rendered_chat = compiled_template.render(
            messages=chat, add_generation_prompt=add_generation_prompt,
        )
        rendered.append(rendered_chat)

    return rendered

# Example usage
if __name__ == '__main__':
    messages = [
        {'role': 'user', 'content': '介绍一下人工智能的发展历程。'}
    ]
    chat_template = default_chat_template()
    rendered = render_chat_template(messages, chat_template)
    print(rendered)
