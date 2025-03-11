import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from ..precision import get_autocast, get_input_dtype
from .metrics import auroc, accuracy_and_f1
import logging


def run(X, y, X_test, y_test):
    _, num_classes = y.shape

    preds = np.zeros_like(y_test)
    for i in range(num_classes):
        lr_model = LogisticRegression()
        lr_model.fit(X, y[:, i])
        preds[:, i] = lr_model.predict_proba(X_test)[:, 1]

    acc, f1, _, _ = accuracy_and_f1(y_test, preds)
    auc, _ = auroc(y_test, preds)
    return acc, f1, auc


def linear_probe_eval(model, train_data, test_data, args, dataset=""):
    logging.info(f'Starting linear-probe {dataset}.')
    metrics = {}
    all_train_ecg_features = []
    all_train_labels = []
    all_test_ecg_features = []
    all_test_labels = []

    device = args.device
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    with torch.no_grad():
        for i, batch in enumerate(train_data):
            ecgs, targets = batch
            all_train_labels.append(targets)
            ecgs = ecgs.to(device=device, dtype=input_dtype, non_blocking=True)
            with autocast():
                output = model(ecg=ecgs)
                ecg_features = output['ecg_features'] if isinstance(output, dict) else output[0]
                all_train_ecg_features.append(ecg_features.cpu())

        all_train_ecg_features = torch.cat(all_train_ecg_features)
        all_train_labels = torch.cat(all_train_labels)
        for i, batch in enumerate(test_data):
            ecgs, targets = batch
            all_test_labels.append(targets)
            ecgs = ecgs.to(device=device, dtype=input_dtype, non_blocking=True)
            with autocast():
                output = model(ecg=ecgs)
                ecg_features = output['ecg_features'] if isinstance(output, dict) else output[0]
                all_test_ecg_features.append(ecg_features.cpu())
        all_test_ecg_features = torch.cat(all_test_ecg_features)
        all_test_labels = torch.cat(all_test_labels)
        acc, f1, roc_auc = run(all_train_ecg_features, all_train_labels, all_test_ecg_features, all_test_labels)
        metrics[f"{dataset}-linear-probe-val-acc"] = acc
        metrics[f"{dataset}-linear-probe-val-f1-score"] = f1
        metrics[f"{dataset}-linear-probe-val-auc"] = roc_auc
        return metrics
