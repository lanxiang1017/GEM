import logging
import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..precision import get_autocast, get_input_dtype
import torch.nn.functional as F
from .metrics import auroc, accuracy_and_f1
from .metadata import zero_shot_class


def build_zero_shot_classifier(args, model, tokenizer, dataset=""):
    texts = zero_shot_class[dataset]
    device = args.device
    with open('./training/evaluation/CKEPE_prompt.json', 'r', encoding='utf-8') as file:
        prompt = json.load(file)
    texts_encoded = []
    for text in texts:
        text = text.replace('_', "").replace("(s)", "")
        texts_encoded.append(tokenizer([prompt[text]])[0])

    texts_encoded = torch.stack(texts_encoded).to(device)

    with torch.no_grad():
        class_embedding = model.encode_text(texts_encoded)
    return class_embedding.T


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    num_sample = dataloader.num_samples
    y_true = np.zeros((num_sample, classifier.shape[1]))
    y_pred = np.zeros_like(y_true)

    i = 0
    with torch.no_grad():
        for ecgs, targets in dataloader:
            batch_size = ecgs.shape[0]
            ecgs = ecgs.to(device=args.device, dtype=input_dtype)
            with autocast():
                # predict
                output = model(ecg=ecgs)
                ecg_features = output['ecg_features'] if isinstance(output, dict) else output[0]
                logits = model.logit_scale.exp() * ecg_features @ classifier

            logits = F.sigmoid(logits).cpu().numpy()
            y_true[i:i+batch_size, :] = targets
            y_pred[i:i+batch_size, :] = logits
            i += batch_size

    acc, f1, _, _ = accuracy_and_f1(y_true, y_pred)
    auc, _ = auroc(y_true, y_pred)
    return acc, f1, auc


def zero_shot_eval(model, data, args, tokenizer, dataset=""):
    logging.info(f'Starting zero-shot {dataset}.')
    assert tokenizer is not None
    autocast = get_autocast(args.precision)
    with autocast():
        classifier = build_zero_shot_classifier(
            args,
            model,
            tokenizer=tokenizer,
            dataset=dataset
        )

    results = {}
    acc, f1, auc = run(model, classifier, data, args)
    results[f'{dataset}-zeroshot-val-acc'] = acc
    results[f'{dataset}-zeroshot-val-f1-score'] = f1
    results[f'{dataset}-zeroshot-val-auc'] = auc
    return results
