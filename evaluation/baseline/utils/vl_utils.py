import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def make_interleave_content(texts_or_image_paths):
    # texts_or_image_paths: ["XXX", "1.jpg", "YYY", "2.jpg", "ZZZ"]
    content = []
    for text_or_path in texts_or_image_paths:
        if text_or_path.startswith("<|image|>"):
            image_path = text_or_path.replace("<|image|>","")
            base64_image = encode_image(image_path)
            image_elem = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low",
                },
            }
            content.append(image_elem)
        else:
            text_elem = {
                "type": "text",
                "text": text_or_path,
            }
            content.append(text_elem)
    return content

def make_interleave_content_dummy(texts_or_image_paths):
    # texts_or_image_paths: ["XXX", "1.jpg", "YYY", "2.jpg", "ZZZ"]
    content = []
    for text_or_path in texts_or_image_paths:
        if text_or_path.startswith("<|image|>"):
            image_path = text_or_path.replace("<|image|>","")
            # base64_image = encode_image(image_path)
            image_elem = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,<base64 image placeholder>",
                    "detail": "low",
                },
            }
            content.append(image_elem)
        else:
            text_elem = {
                "type": "text",
                "text": text_or_path,
            }
            content.append(text_elem)
    return content
