import os
import wfdb
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from .evaluation.metadata import zero_shot_class
from .distributed import is_master

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class ECGTextDataset(Dataset):
    def __init__(self, path, texts, transforms=None, tokenizer=None, is_train=True):
        super(ECGTextDataset, self).__init__()
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.path = path
        self.y = texts
        self.is_train = is_train

    def tokenize(self, text):
        text = text.lower()
        encoded = self.tokenizer(
            text,
        )
        return encoded[0]

    def load_data(self, idx):
        data = wfdb.rdsamp(self.path[idx])[0]
        data[np.isnan(data)] = 0
        data[np.isinf(data)] = 0
        data = torch.Tensor(np.transpose(data, (1, 0)).astype(np.float32))
        data = torch.unsqueeze(data, 0)

        if self.transforms is not None:
            data = self.transforms(data)
        data = torch.squeeze(data, 0)
        return data

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.load_data(idx)
        y = self.y[idx]
        return x, self.tokenize(y)


class ECGValDataset(ECGTextDataset):
    def __init__(self, dir, path, diagnostics, transforms=None, tokenizer=None):
        abs_path = [os.path.join(dir, p) for p in path]
        super(ECGValDataset, self).__init__(abs_path, None, transforms, tokenizer)
        self.diagnostics = diagnostics

    def __len__(self):
        return self.diagnostics.shape[0]

    def __getitem__(self, idx):
        x = self.load_data(idx)
        diagnostic = self.diagnostics[idx, :]
        return x, diagnostic


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


def get_wave_info(data):
    keys = ['RR_Interval', 'PR_Interval', 'QRS_Complex', 'QT_Interval',
            'QTc_Interval', 'P_Wave_Peak', 'R_Wave_Peak', 'T_Wave_Peak']
    text_describe = ""
    text_describe += f" RR: {data['RR_Interval']}"
    text_describe += f" PR: {data['PR_Interval']}"
    text_describe += f" QRS: {data['QRS_Complex']}"
    text_describe += f" QT/QTc: {data['QT_Interval']}/{data['QTc_Interval']}"
    text_describe += f" P/R/T Wave: {data['P_Wave_Peak']}/{data['R_Wave_Peak']}/{data['T_Wave_Peak']}"
    return text_describe


def load_ptbxl(path, is_train, sampling_rate=500, wfep=True):
    Y = pd.read_csv(os.path.join(path, 'ptbxl_database_translated.csv'), index_col='ecg_id')

    test_fold = 10
    if is_train:
        Y = Y[Y.strat_fold != test_fold]
    else:
        Y = Y[Y.strat_fold == test_fold]

    if sampling_rate == 500:
        X_rel = Y.filename_hr.values
    else:
        X_rel = Y.filename_lr.values
    y = Y.report.values
    X = [os.path.join(path, x) for x in X_rel]

    texts = []
    for i in range(len(Y)):
        text = (y[i].replace("ekg", "ecg")
                .replace("normales ", "")
                .replace("4.46 ", "")
                .replace("unconfirmed report", "")
                .replace("unconfirmed", "").replace(" age undetermined", ""))
        if wfep:
            texts.append(text + get_wave_info(Y.iloc[i]))
        else:
            texts.append(text)

    return X, texts


def load_ptbxl_diagnostics(path, is_train, sampling_rate=500):
    dataset_names = ["super_class", "sub_class", "form", "rhythm"]
    data = {}
    for dataset in dataset_names:
        name2index = {}
        for i, name in enumerate(zero_shot_class["ptbxl_" + dataset]):
            name2index[name] = i
        Y = pd.read_csv(os.path.join(path, f"ptbxl_database_{dataset}.csv"))

        test_fold = 10
        if is_train:
            Y = Y[Y.strat_fold != test_fold]
        else:
            Y = Y[Y.strat_fold == test_fold]

        if sampling_rate == 500:
            X_rel = Y.filename_hr.values
        else:
            X_rel = Y.filename_lr.values
        X = [os.path.join(path, x) for x in X_rel]
        y = Y.labels.values

        labels = [label.split(';') for label in y]

        targets = np.zeros((len(X), len(zero_shot_class["ptbxl_" + dataset])))
        for i in range(len(X)):
            for lbl in labels[i]:
                targets[i][name2index[lbl]] = 1

        data[dataset] = (X, targets)
    return data


def load_cpsc2018(path, is_train):
    folder = "training_set" if is_train else "validation_set"
    Y = pd.read_csv(os.path.join(path, folder, "REFERENCE.csv"))
    X = Y.Recording.values
    X = [os.path.join(path, folder + "/" + x) for x in X]
    Y.fillna(10000)
    lbl1 = Y.First_label.values.astype(int)
    lbl2 = Y.Second_label.values.astype(int)
    lbl3 = Y.Third_label.values.astype(int)
    labels = np.zeros((len(X), 9))
    for i in range(len(X)):
        labels[i, lbl1[i] - 1] = 1
        if 0 < lbl2[i] < 10:
            labels[i, lbl2[i] - 1] = 1
        if 0 < lbl3[i] < 10:
            labels[i, lbl3[i] - 1] = 1
    return X, labels


def load_champan_shaoxing(path, is_train, wfep=True):
    Y = pd.read_csv(os.path.join(path, "diagnostics.csv"))

    test_fold = 9

    if is_train:
        Y = Y[Y.strat_fold != test_fold]
    else:
        Y = Y[Y.strat_fold == test_fold]

    X_rel = Y.filename.values
    X = [os.path.join(path, x) for x in X_rel]
    y = Y.report.values

    if wfep:
        for i in range(len(Y)):
            y[i] = y[i] + get_wave_info(Y.iloc[i])
    return X, y


def load_sph(path, is_train, wfep=False):
    df = pd.read_csv(os.path.join(path, "metadata.csv"))

    # 80%-20% split data accroding to the example code
    test1 = df.Patient_ID.duplicated(keep=False)
    N = int(len(df) * 0.2) - sum(test1)
    # 73 is chosen such that all primary statements exist in both sets
    df_test = pd.concat([df[test1], df[~test1].sample(N, random_state=73)])
    df_train = df.iloc[df.index.difference(df_test.index)]

    Y = df_train if is_train else df_test
    X = Y.ECG_ID.values
    X = [os.path.join(path, "records/records_wfdb/" + x) for x in X]
    aha_codes = Y.AHA_Code.values

    code_csv = pd.read_csv(os.path.join(path, "code.csv"))
    code2text = {}
    for i in range(len(code_csv)):
        code2text[str(code_csv.Code.values[i])] = code_csv.Description.values[i]

    texts = []
    for codes in aha_codes:
        code_list = codes.split(';')
        text_list = []
        for code in code_list:
            t = " ".join(code2text[c] for c in code.split('+'))
            text_list.append(t)
        texts.append(";".join(text_list))
    if wfep:
        for i in range(len(Y)):
            texts[i] = texts[i] + get_wave_info(Y.iloc[i])

    return X, texts


def load_mimic_iv_ecg(path, wfep=True):
    database = pd.read_csv(os.path.join(path, "machine_measurements.csv")).set_index("study_id")
    record_list = pd.read_csv(os.path.join(path, "new_record_list.csv"))

    indexes = record_list.index.values
    np.random.seed(0)
    np.random.shuffle(indexes)

    train_list = record_list.loc[np.where(record_list["file_name"].values % 10 > 0)].set_index("study_id")
    test_list = record_list.loc[np.where(record_list["file_name"].values % 10 == 0)].set_index("study_id")
    train_indexes = train_list.index.values
    val_indexes = test_list.index.values[-10000:-5000]
    test_indexes = test_list.index.values[-5000:]
    train_indexes = np.append(train_indexes, test_indexes[:-10000])

    def data(index_list):
        reports = []
        X = []
        n_reports = 18
        bad_reports = ["--- Warning: Data quality may affect interpretation ---",
                       "--- Recording unsuitable for analysis - please repeat ---",
                       "Analysis error",
                       "conduction defect",
                       "*** report made without knowing patient's sex ***",
                       "--- Suspect arm lead reversal",
                       "--- Possible measurement error ---",
                       "--- Pediatric criteria used ---",
                       "--- Suspect limb lead reversal",
                       "-------------------- Pediatric ECG interpretation --------------------",
                       "Lead(s) unsuitable for analysis:",
                       "LEAD(S) UNSUITABLE FOR ANALYSIS:",
                       "PACER DETECTION SUSPENDED DUE TO EXTERNAL NOISE-REVIEW ADVISED",
                       "Pacer detection suspended due to external noise-REVIEW ADVISED"]

        for i in index_list:
            row = record_list.loc[i]
            m_row = database.loc[i]
            report_txt = ""
            for j in range(n_reports):
                report = m_row[f"report_{j}"]
                if type(report) == str:
                    is_bad = False
                    for bad_report in bad_reports:
                        if report.find(bad_report) > -1:
                            is_bad = True
                            break
                    report_txt += (report + " ") if not is_bad else ""
            if report_txt == "":
                continue
            report_txt = report_txt[:-1].lower()
            report_txt = (report_txt.replace("---", "")
                          .replace("***", "")
                          .replace(" - age undetermined", ""))

            report_txt = (report_txt.replace('rbbb', 'right bundle branch block')
                          .replace('lbbb', 'light bundle branch block')
                          .replace('lvh', 'left ventricle hypertrophy')
                          .replace("mi", "myocardial infarction")
                          .replace("lafb", "left anterior fascicular block")
                          .replace("pvc(s)", "ventricular premature complex")
                          .replace("pvcs", "ventricular premature complex")
                          .replace("pac(s)", "atrial premature complex")
                          .replace("pacs", "atrial premature complex"))
            if wfep:
                report_txt = report_txt + get_wave_info(row)
            reports.append(report_txt)
            X.append(os.path.join(path, row["path"]))
        return X, reports

    record_list = record_list.set_index("study_id")
    train_x, train_y = data(train_indexes)

    val_x, val_y = data(val_indexes)
    test_x, test_y = data(test_indexes)
    return train_x, train_y, val_x, val_y, test_x, test_y


def make_dataloader(args, dataset, is_train, dist_sampler=True, drop_last=None):
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and dist_sampler else None
    shuffle = is_train and sampler is None
    drop_last = is_train if drop_last is None else drop_last
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=drop_last,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_ptbxl_diagnostic_dataset(args, path, preprocess_fn, is_train, epoch=0, tokenizer=None):
    sampling_rate = args.sampling_rate if args.sampling_rate is not None else 500
    datas = load_ptbxl_diagnostics(path, is_train, sampling_rate)

    dataloaders = {}
    for key in datas.keys():
        X, y = datas[key]
        dataset = ECGValDataset(path, X, y, transforms=preprocess_fn, tokenizer=tokenizer)
        dataloaders[key] = make_dataloader(args, dataset, is_train, dist_sampler=False, drop_last=False)
    return dataloaders


def get_cpsc2018_diagnostic_dataset(args, path, preprocess_fn, is_train, epoch=0, tokenizer=None):
    X, y = load_cpsc2018(path, is_train)
    dataset = ECGValDataset(path, X, y, transforms=preprocess_fn, tokenizer=tokenizer)

    return make_dataloader(args, dataset, is_train, dist_sampler=False, drop_last=False)


def get_all_ecg_text_dataset(args, preprocess_train, preprocess_test, epoch=0, tokenizer=None):
    datasets = {}
    X_train, text_train, X_val, text_val, m_X_test, m_text_test\
        = load_mimic_iv_ecg(args.mimic_iv_ecg_path, wfep=args.wfep)
    if args.champan_path:
        c_X_train, c_text_train = load_champan_shaoxing(args.champan_path, is_train=True, wfep=args.wfep)
        X_train = np.append(X_train, c_X_train)
        text_train = np.append(text_train, c_text_train)
    if args.sph_path:
        s_X_train, s_text_train = load_sph(args.sph_path, is_train=True, wfep=args.wfep)
        X_train = np.append(X_train, s_X_train)
        text_train = np.append(text_train, s_text_train)
    train_dataset = ECGTextDataset(X_train, text_train, transforms=preprocess_train, tokenizer=tokenizer,
                                   is_train=True)
    datasets['train'] = make_dataloader(args, train_dataset, is_train=True)

    if is_master(args):
        # if args.champan_path:
        #     c_X_val, c_text_val = load_champan_shaoxing(args.champan_path, is_train=False)
        #     # X_val = np.append(X_val, c_X_val)
        #     # text_val = np.append(text_val, c_text_val)
        #
        # if args.sph_path:
        #     s_X_val, s_text_val = load_sph(args.sph_path, is_train=False)
        #     X_val = np.append(X_val, s_X_val)
        #     text_val = np.append(text_val, s_text_val)

        val_dataset = ECGTextDataset(X_val, text_val, transforms=preprocess_test, tokenizer=tokenizer, is_train=False)
        mimic_test_dataset = ECGTextDataset(m_X_test, m_text_test, transforms=preprocess_test,
                                            tokenizer=tokenizer, is_train=False)
        datasets['val'] = make_dataloader(args, val_dataset, is_train=False, dist_sampler=False)
        datasets['test_mimic'] = make_dataloader(args, mimic_test_dataset, is_train=False, dist_sampler=False)

        if args.ptbxl_path:
            p_X_test, p_text_test = load_ptbxl(args.ptbxl_path, is_train=False, wfep=args.wfep)
            ptbxl_test_dataset = ECGTextDataset(p_X_test, p_text_test, transforms=preprocess_test,
                                                tokenizer=tokenizer, is_train=False)
            datasets['test_ptbxl'] = make_dataloader(args, ptbxl_test_dataset, is_train=False, dist_sampler=False)
    return datasets


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns

    data = get_all_ecg_text_dataset(args, preprocess_train, preprocess_val, epoch=epoch, tokenizer=tokenizer)
    if is_master(args):
        if args.ptbxl_path:
            ptbxl_train_datas = get_ptbxl_diagnostic_dataset(
                args, args.ptbxl_path, preprocess_val, is_train=True, epoch=epoch, tokenizer=tokenizer)
            ptbxl_test_datas = get_ptbxl_diagnostic_dataset(
                args, args.ptbxl_path, preprocess_val, is_train=False, epoch=epoch, tokenizer=tokenizer)

            for key in ptbxl_train_datas.keys():
                data[f"train_ptbxl_{key}"] = ptbxl_train_datas[key]
                data[f"val_ptbxl_{key}"] = ptbxl_test_datas[key]
        if args.cpsc2018_path:
            data["train_cpsc2018"] = get_cpsc2018_diagnostic_dataset(
                args, args.cpsc2018_path, preprocess_val, is_train=True, epoch=epoch, tokenizer=tokenizer)
            data["val_cpsc2018"] = get_cpsc2018_diagnostic_dataset(
                args, args.cpsc2018_path, preprocess_val, is_train=False, epoch=epoch, tokenizer=tokenizer)

    return data
