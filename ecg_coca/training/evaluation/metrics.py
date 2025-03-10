import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, \
    precision_recall_curve


def accuracy_and_f1(y_true, y_pred):
    target_class = y_true.shape[1]
    accs = []
    max_f1s = []
    for i in range(target_class):
        gt_np = y_true[:, i]
        pred_np = y_pred[:, i]
        precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
        numerator = 2 * recall * precision
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
        max_f1 = np.max(f1_scores)
        max_f1_thresh = thresholds[np.argmax(f1_scores)]
        max_f1s.append(max_f1)
        accs.append(accuracy_score(gt_np, pred_np > max_f1_thresh))

    max_f1s = [i * 100 for i in max_f1s]
    accs = [i * 100 for i in accs]
    f1_avg = np.array(max_f1s).mean()
    acc_avg = np.array(accs).mean()

    return acc_avg, f1_avg, accs, max_f1s


def auroc(y_true, y_pred):
    AUROCs = []
    gt_np = y_true
    pred_np = y_pred
    n_class = y_true.shape[1]
    for i in range(n_class):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i], average='macro', multi_class='ovo'))
    AUROCs = [i * 100 for i in AUROCs]
    AUROC_avg = np.array(AUROCs).mean()
    return AUROC_avg, AUROCs
