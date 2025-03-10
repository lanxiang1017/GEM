import os
import json
import random
random.seed(42)
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score,hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer

def extract(text):
    if text in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        return text
    for char_dot in ["A.", "B.", "C.", "D.", "E.", "F.", "G.", "H."]:
        if char_dot in text:
            return char_dot[0]
    if "The correct option is " in text:
        predict_char = text.split("The correct option is ")[-1][0]
        if predict_char in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            return predict_char
        else:
            return None
    if "Answer:" in text:
        answer = text.split("Answer:")[-1].strip()
        # if answer in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        #   return answer
        for char in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            if char in answer:
                return char
        else:
            return None
    else:
        return None
    
def compute_f1_auc(y_pred, y_true):
    # Binarize labels
    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    # print(y_true)
    # print(y_true_bin)
    hl = hamming_loss(y_true_bin, y_pred_bin)
    
    f1_scores_all = []
    # Compute the F1 score
    f1_scores = f1_score(y_true_bin, y_pred_bin, average=None)
    for idx, cls in enumerate(mlb.classes_):
        # print(f'F1 score for class {cls}: {f1_scores[idx]}')
        f1_scores_all.append(f1_scores[idx])
    
    # Compute the AUC score
    auc_scores = []
    for i in range(y_true_bin.shape[1]):
        try:
            auc = roc_auc_score(y_true_bin[:, i], y_pred_bin[:, i])
        except ValueError:
            auc = np.nan    # If AUC cannot be calculated, NaN is returned
        auc_scores.append(auc)
        # print(f'AUC score for class {mlb.classes_[i]}: {auc}')    
    # print("f1 all",np.mean(f1_scores_all), "auc all", np.mean(auc_scores))
    return np.mean(f1_scores_all), np.mean(auc_scores), hl
    
def eval_mmmu(dir):
    print("====mmmu====")
    with open("", "r", encoding='utf-8') as f:
        data = json.load(f)
        answer_dict = {item["No"]: item["conversations"][1]["value"] for item in data}
    
    score_dict = {}
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".jsonl"):
                predict_list = []
                golden_list = []
                file_path = os.path.join(root, file)
                if "mmmu-ecg" not in file_path:
                    continue
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        if "question_id" in item:
                            qid = item["question_id"]
                        elif "id" in item:
                            qid = item["id"]
                        if "text" in item:
                            predict = extract(item["text"])
                        elif "response" in item:
                            predict = extract(item["response"])
                        if predict is None:
                            predict = random.choice(["A", "B", "C", "D"])
                        # print(predict)
                        golden = answer_dict[qid]
                        predict_list.append(predict)
                        golden_list.append(golden)
                # print(predict_list)
                if len(predict_list) != 200:
                    continue

                accuracy = accuracy_score(golden_list, predict_list)
                
                print(file, f"Accuracy: {accuracy}")
                if "step" in file:
                    if file.split("-")[-1].split(".")[0] == "final":
                        step_num = 99999
                    else:
                        step_num = int(file.split("-")[-1].split(".")[0])
                else:
                    step_num = file.split("_")[0]
                # step_num = int(file.split("-")[-1].split(".")[0])
                score_dict[step_num] = accuracy
    for step_num in sorted(score_dict):
        if step_num == 99999:
            print(f"Model final Accuracy: {score_dict[step_num]:.4f}")
        else:
            print(f"Model {step_num} Accuracy: {score_dict[step_num]:.4f}")


def eval_ptb_test(dir):
    print("====ptb test====")
    label_space = ["NORM","MI","STTC","CD","HYP"]
    golden_data_path = ""
    golden_label = {}
    with open(golden_data_path, "r", encoding='utf-8') as f:
        golden_data = json.load(f)
        for item in golden_data:
            qid = item["id"]
            golden_label[qid] = [label for label in label_space if label in item["conversations"][1]["value"]]
        
    score_dict = {}
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                if "ptb-test" not in file_path or "report" in file_path:
                    continue
                predict_list = []
                golden_list = []
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        if "question_id" in item:
                            qid = item["question_id"]
                        elif "id" in item:
                            qid = item["id"]
                        if "text" in item:
                            predict = [label for label in label_space if label in item["text"]]
                        elif "response" in item:
                            predict = [label for label in label_space if label in item["response"]]
                        
                        true = golden_label[qid]
                        predict_list.append(predict)
                        golden_list.append(true)
                f1, auc, hl = compute_f1_auc(predict_list, golden_list)
                print(file, "f1", round(f1*100, 1), "auc", round(auc*100, 1), "hl", round(hl*100, 1))
                
                if "step" in file:
                    if file.split("-")[-1].split(".")[0] == "final":
                        step_num = 99999
                    else:
                        step_num = int(file.split("-")[-1].split(".")[0])
                else:
                    step_num = file.split("_")[0]
                # step_num = int(file.split("-")[-1].split(".")[0])
                score_dict[step_num] = f"F1: {f1:.4f}, AUC: {auc:.4f}"
    for step_num in sorted(score_dict):
        if step_num == 99999:
            print(f"Model final Accuracy: {score_dict[step_num]}")
        else:
            print(f"Model {step_num} Accuracy: {score_dict[step_num]}")


                
def eval_cpsc_test(dir):
    print("====cpsc test====")
    label_space = ["NORM", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
    golden_data_path = ""
    golden_label = {}
    with open(golden_data_path, "r", encoding='utf-8') as f:
        golden_data = json.load(f)
        for item in golden_data:
            qid = item["id"]
            golden_label[qid] = [label for label in label_space if label in item["conversations"][1]["value"]]
    # print(golden_label)
    score_dict = {}
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                if "cpsc-test" not in file_path:
                    continue
                predict_list = []
                golden_list = []
                # print(file,"=====")
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        if "question_id" in item:
                            qid = item["question_id"]
                        elif "id" in item:
                            qid = item["id"]
                        if "text" in item:
                            predict = [label for label in label_space if label in item["text"]]
                        elif "response" in item:
                            predict = [label for label in label_space if label in item["response"]]

                        true = golden_label[qid]

                    
                        predict_list.append(predict)
                       
                        golden_list.append(true)
                        
                f1, auc, hl = compute_f1_auc(predict_list, golden_list)
                print(file, "f1", round(f1*100, 1), "auc", round(auc*100, 1), "hl", round(hl*100, 1))
                if "step" in file:
                    if file.split("-")[-1].split(".")[0] == "final":
                        step_num = 99999
                    else:
                        step_num = int(file.split("-")[-1].split(".")[0])
                else:
                    step_num = file.split("_")[0]
                # step_num = int(file.split("-")[-1].split(".")[0])
                score_dict[step_num] = f"F1: {f1:.4f}, AUC: {auc:.4f}"
    for step_num in sorted(score_dict):
        if step_num == 99999:
            print(f"Model final Accuracy: {score_dict[step_num]}")
        else:
            print(f"Model {step_num} Accuracy: {score_dict[step_num]}")

        
def eval_ecgqa_test(dir):
    print("====ecgqa test====")
    golden_data_path = "" # your path to ecgqa test
    golden_label = {}
    with open(golden_data_path, "r", encoding='utf-8') as f:
        golden_data = json.load(f)
        for item in golden_data:
            qid = item["id"]
            golden_label[qid] = item["conversations"][1]["value"]
            
    score_dict = {}
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                if "ecgqa-test" not in file_path:
                    continue
                
                pass_list = []
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        if "question_id" in item:
                            qid = item["question_id"]
                        elif "id" in item:
                            qid = item["id"]
                        
                        if "prompt" in item:
                            if isinstance(item["prompt"], dict):
                                candidates = [i.strip() for i in item["prompt"]["prompt"].split("Options:")[-1].replace("Only answer based on the given Options without any explanation.","").split(",")]
                            else:
                                candidates = [i.strip() for i in item["prompt"].split("Options:")[-1].replace("Only answer based on the given Options without any explanation.","").split(",")]
                        
                        if "text" in item:
                            predict = [i for i in candidates if i.lower() in item["text"].lower()]
                        elif "response" in item:
                            predict = [i for i in candidates if i in item["response"].lower()]
                        if isinstance(predict, list):
                            predict_str = ''.join(predict)
                        else:
                            predict_str = predict
                        if set(predict_str) == set(golden_label[qid]):
                            pass_list.append(1)
                        else:
                            pass_list.append(0)
                accuracy = sum(pass_list) / len(pass_list)
                print(file,"accuracy",accuracy)
                if "step" in file:
                    if file.split("-")[-1].split(".")[0] == "final":
                        step_num = 99999
                    else:
                        step_num = int(file.split("-")[-1].split(".")[0])
                else:
                    step_num = file.split("_")[0]
                
                score_dict[step_num] = accuracy
    for step_num in sorted(score_dict):
        if step_num == 99999:
            print(f"Model final Accuracy: {score_dict[step_num]:.4f}")
        else:
            print(f"Model {step_num} Accuracy: {score_dict[step_num]:.4f}")
                            
def eval_code15_test(dir):
    print("====code15 test====")
    label_space = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]
    golden_data_path = ""
    golden_label = {}
    with open(golden_data_path, "r", encoding='utf-8') as f:
        golden_data = json.load(f)
        for item in golden_data:
            qid = item["id"]
            if item["conversations"][1]["value"] == "NORM":
                golden_label[qid] = ["NORM"]
            elif item["conversations"][1]["value"] == "ABNORMAL":
                golden_label[qid] = ["ABNORMAL"]
            else:
                golden_label[qid] = [label for label in label_space if label in item["conversations"][1]["value"]]
        
    score_dict = {}
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                if "code15-test" not in file_path:
                    continue
                predict_list = []
                golden_list = []
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        if "question_id" in item:
                            qid = item["question_id"]
                        elif "id" in item:
                            qid = item["id"]
                        if "text" in item:
                            if "Answer:" in item["text"]:
                                item["text"] = item["text"].split("Answer:")[-1]
                            if "NORM" in item["text"] and "ABNORMAL" not in item["text"]:
                                predict = ["NORM"] + [label for label in label_space if label in item["text"]]
                            elif "ABNORMAL" in item["text"]:
                                predict = ["ABNORMAL"] + [label for label in label_space if label in item["text"]]
                            else:
                                predict = [label for label in label_space if label in item["text"]]
                        elif "response" in item:
                            if "Answer:" in item["response"]:
                                item["response"] = item["response"].split("Answer:")[-1]
                            if "NORM" in item["response"] and "ABNORMAL" not in item["response"]:
                                predict = ["NORM"] + [label for label in label_space if label in item["response"]]
                            elif "ABNORMAL" in item["response"]:
                                predict = ["ABNORMAL"] + [label for label in label_space if label in item["response"]]
                            else:
                                predict = [label for label in label_space if label in item["response"]]
                    
                        true = golden_label[qid]
                        predict_list.append(predict)
                        golden_list.append(true)
                f1, auc, hl = compute_f1_auc(predict_list, golden_list)
                print(file, "f1", round(f1*100, 1), "auc", round(auc*100, 1), "hl", round(hl*100, 1))
                if "step" in file:
                    if file.split("-")[-1].split(".")[0] == "final":
                        step_num = 99999
                    else:
                        step_num = int(file.split("-")[-1].split(".")[0])
                else:
                    step_num = file.split("_")[0]
                # step_num = int(file.split("-")[-1].split(".")[0])
                score_dict[step_num] = f"F1: {f1:.4f}, AUC: {auc:.4f}"
    for step_num in sorted(score_dict):
        if step_num == 99999:
            print(f"Model final Accuracy: {score_dict[step_num]}")
        else:
            print(f"Model {step_num} Accuracy: {score_dict[step_num]}")

        
def eval_csn_test(dir):
    print("====csn test====")
    with open("", "r", encoding='utf-8') as f:
        data = json.load(f)
        answer_dict = {item["id"]: item["conversations"][1]["value"][0] for item in data}
    
    score_dict = {}
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                if "csn-test" not in file_path:
                    continue
            
                predict_list = []
                golden_list = []
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        if "question_id" in item:
                            qid = item["question_id"]
                        elif "id" in item:
                            qid = item["id"]
                        if "text" in item:
                            predict = extract(item["text"])
                        elif "response" in item:
                            predict = extract(item["response"])
                        if predict is None:
                            # print(file)
                            # print(item)
                            predict = random.choice(["A", "B", "C", "D", "E", "F", "G", "H"])
                        
                        golden = answer_dict[qid]
                        predict_list.append(predict)
                        golden_list.append(golden)
                
                if len(predict_list) != 1611:
                    continue

                accuracy = accuracy_score(golden_list, predict_list)
                print(file, f"Accuracy: {accuracy}")
                # print(file, f"Accuracy: {accuracy}")
                if "step" in file:
                    if file.split("-")[-1].split(".")[0] == "final":
                        step_num = 99999
                    else:
                        step_num = int(file.split("-")[-1].split(".")[0])
                else:
                    step_num = file.split("_")[0]
                # step_num = int(file.split("-")[-1].split(".")[0])
                score_dict[step_num] = accuracy
    for step_num in sorted(score_dict):
        if step_num == 99999:
            print(f"Model final Accuracy: {score_dict[step_num]:.4f}")
        else:
            print(f"Model {step_num} Accuracy: {score_dict[step_num]:.4f}")

        
def eval_g12_test(dir):
    print("====g12 test====")
    with open("", "r", encoding='utf-8') as f:
        data = json.load(f)
        answer_dict = {item["id"]: item["conversations"][1]["value"][0] for item in data}
    
    score_dict = {}
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                if "g12-test" not in file_path:
                    continue
                predict_list = []
                golden_list = []
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        if "question_id" in item:
                            qid = item["question_id"]
                        elif "id" in item:
                            qid = item["id"]
                        if "text" in item:
                            predict = extract(item["text"])
                        elif "response" in item:
                            predict = extract(item["response"])
                        if predict is None:
                            
                            predict = random.choice(["A", "B", "C", "D", "E", "F", "G", "H"])
                        
                        golden = answer_dict[qid]
                        predict_list.append(predict)
                        golden_list.append(golden)
                
                if len(predict_list) != 2026:
                    continue

                accuracy = accuracy_score(golden_list, predict_list)
                print(file, f"Accuracy: {accuracy}")
                # print(file, f"Accuracy: {accuracy}")
                if "step" in file:
                    if file.split("-")[-1].split(".")[0] == "final":
                        step_num = 99999
                    else:
                        step_num = int(file.split("-")[-1].split(".")[0])
                else:
                    step_num = file.split("_")[0]
                # step_num = int(file.split("-")[-1].split(".")[0])
                score_dict[step_num] = accuracy
    for step_num in sorted(score_dict):
        if step_num == 99999:
            print(f"Model final Accuracy: {score_dict[step_num]:.4f}")
        else:
            print(f"Model {step_num} Accuracy: {score_dict[step_num]:.4f}")

            
                            
if __name__ == "__main__":
    root = ""
    
    eval_ptb_test(f"{root}/ptb-test")
    eval_cpsc_test(f"{root}/cpsc-test")
    eval_csn_test(f"{root}/csn-test-no-cot")
    eval_g12_test(f"{root}/g12-test-no-cot")
    eval_code15_test(f"{root}/code15-test")
    eval_ecgqa_test(f"{root}/ecgqa-test")