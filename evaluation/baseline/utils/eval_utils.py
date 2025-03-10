import json
import re
import sympy as sp

idx_ranges = [
    list(range(211, 221)),  
    [18],                   
    [73, 74],               
    [94],                    
    [115, 116, 117],         
    [141],                   
    [143],                  
    list(range(145, 150)),  
    [153, 154, 155, 156],   
    [165]                    
]

def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def extract_text_from_brackets(text):
    matches = re.findall(r'\[\[\s*(.*?)\s*\]\]', text, re.DOTALL)
    if matches:
        first_match = matches[0].strip().replace('"', '').replace(' ', '')
        return f'[[{first_match}]]'
    return "NULL"

def extract_text_from_brackets_puzzle(text):
    if not isinstance(text, str):
        print(f"text type: {type(text)}, text value: {text}")
        return "NULL"
    match = re.search(r'\[\[(.*?)\]\]', text, re.DOTALL)
    return match.group(1) if match else "NULL"
def extract_numbers_from_latex(latex_str):
    numbers = re.findall(r'\d+', latex_str)
    numbers = list(map(int, numbers))
    return numbers

def rule5_extract_text(text):
    matches = re.findall(r'\[\[(.*?)\]\]', text)
    extracted_text = matches[0].replace(' ', '') if matches else None
    if extracted_text:
        return f'[{extracted_text}]'
    return None

def rule5_normalize_content(content):
    parts = [part for part in content.split(';')]
    sorted_parts = sorted(parts)
    return sorted_parts

def normalize_string(s):
    s = re.sub(r'[^0-9]', '', s)
    pairs = s.split(",")
    pairs.sort()
    return pairs

def remove_commas_and_spaces(s):
    return re.sub(r'[,\s\[\]]+', '', s)

def remove_non_alphanumeric(s):
    return re.sub(r'\W+', '', s)

def method_equal(response_text, answer):
    return response_text==answer

def method_1(response_text, answer):
    cleaned_string = re.sub(r'[^A-Za-z]', '', response_text)
    cleaned_string = cleaned_string.lower()
    answer=re.sub(r'[^A-Za-z]', '', answer)
    answer= answer.lower()
    return cleaned_string == answer

def method_2(response_text, answer):
    cleaned_string = re.sub(r'[^A-Za-z]', '', response_text)
    cleaned_string = cleaned_string.lower()
    answer=answer.split(",")
    return cleaned_string in answer

def method_3(response_text, answer):
    response_text = response_text.lower()
    pairs1=response_text.split(" ")
    pairs2=answer.split(" ")
    pairs1.sort()
    pairs2.sort()
    return pairs1==pairs2

def method_4(response_text, answer):
    cleaned_string = re.sub(r'[^A-Za-z]', '', response_text)
    cleaned_string = cleaned_string.lower()
    return cleaned_string in answer

def method_5(response_text, answer):
    response_text=re.sub(r'\s+', '', response_text)
    response_text=response_text.split(",")
    answer=answer.split(",")
    response_text.sort()
    answer.sort()
    return response_text == answer

def method_9(response_text, answer):

    response_text = response_text.replace('×', '*').replace('−', '-')
    answer = answer.replace('×', '*').replace('−', '-')
    def extract_operators(s):
        return re.findall(r'[+\-*/]', s)
    response_ops = extract_operators(response_text.split('=')[0])
    answer_ops = extract_operators(answer.split('=')[0])
    if response_ops != answer_ops:
        return False
    match = re.search(r'=\s*(-?\d+)', answer)
    expected_result = int(match.group(1))
    try:
        left_side = response_text.split('=')[0]
        result = eval(left_side)
    except Exception as e:
        return False
    return result == expected_result

def method_10(response_text, answer):
    response_text = response_text.replace('×', '*').replace('−', '-')
    response_text=response_text.split('=')[0]
    answer=answer.split('\n')[0].split('=')[0]
    response_ops = sorted(remove_non_alphanumeric(response_text))
    answer_ops = sorted(remove_non_alphanumeric(answer))
    if response_ops != answer_ops:
        return False
    try:
        result = eval(response_text)
    except Exception as e:
        return False
    return result==24

def method_18(response_text, answer):
    cleaned_s1 = remove_commas_and_spaces(response_text)
    cleaned_s2 = remove_commas_and_spaces(answer)
    return cleaned_s1 == cleaned_s2

def method_general(response_text, answer):
    cleaned_s1 = remove_non_alphanumeric(response_text)
    cleaned_s2 = remove_non_alphanumeric(answer)
    return cleaned_s1 == cleaned_s2

question_methods = {
    '1':method_1,
    '2':method_2,
    '3': method_3,
    '4':method_4, 
    '5': method_5, 
    '9':method_9, 
    '10': method_10, 
    '18':method_18,
}

def evaluate_response_vs_answer(response, answer, question_type, rule_id, idx):
    if question_type == 'logic' and rule_id == '5':
        response_text = rule5_extract_text(response)
        answer_text = rule5_extract_text(answer)
        if response_text is None or answer_text is None:
            return False
        normalized_response = rule5_normalize_content(response_text)
        normalized_answer = rule5_normalize_content(answer_text)
        return normalized_response == normalized_answer
    elif question_type == 'operation' and rule_id =='18':
        response_text = extract_text_from_brackets(response)
        answer=extract_text_from_brackets_puzzle(answer)
        response_text = ''.join(sorted(re.sub(r'\W+', '', response_text)))
        answer = ''.join(sorted(re.sub(r'\W+', '', answer)))
        return response_text==answer
    elif question_type == 'operation' and rule_id in {'23', '24', '25'}:
        response_text = extract_text_from_brackets(response)
        if response_text is None:
            return False
        response_text = extract_numbers_from_latex(response_text)
        answer_text =extract_numbers_from_latex(answer)
        return response_text == answer_text
    elif question_type == 'operation' and any(idx in r for r in idx_ranges):
        response_text = extract_text_from_brackets(response)
        response_text_expr = sp.sympify(response_text, evaluate=False)
        answer_expr = sp.sympify(answer, evaluate=False)
        response_text_simplified = sp.simplify(response_text_expr)
        answer_simplified = sp.simplify(answer_expr)
        return response_text_simplified == answer_simplified
    elif question_type == 'puzzle':
        response_text = extract_text_from_brackets_puzzle(response)
        answer=extract_text_from_brackets_puzzle(answer)
        method = question_methods.get(rule_id)
        if method:
            return method(response_text, answer)
        return method_general(response_text,answer)
    else:
        response_text = extract_text_from_brackets(response)
        return response_text == answer
    
def evaluate_responses(data, question_type):
    results = []
    for record in data:
        idx = record.get("idx")
        response = record.get("response")
        response_text = extract_text_from_brackets(response)
        answer = record.get("answer")
        rule_id = record.get("rule_id")
        is_correct = evaluate_response_vs_answer(response, answer, question_type, rule_id, idx)
        result_dict = {
            "idx": idx,
            "response": response,
            "response_text": response_text,
            "answer": answer,
            "is_correct": is_correct
        }
        if question_type == "counterfactual":
            real_life_answer = record.get("real_life_answer")
            is_real_life = evaluate_response_vs_answer(response, real_life_answer, question_type, rule_id, idx)
            result_dict["real_life_answer"] = real_life_answer
            result_dict["is_real_life"] = is_real_life
        results.append(result_dict)
    return results
