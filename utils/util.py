import json
from typing import Dict

import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import itertools

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)

        if key in self._data.index:
            self._data.loc[key, 'total'] += value * n
            self._data.loc[key, 'counts'] += n
            self._data.loc[key, 'average'] = self._data.total[key] / self._data.counts[key]
        else:
            self._data.loc[key] = [value * n, n, value * n]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def count_steps(self, key=None):
        """
        Return the number of updates from the last reset
        :return: int
        """
        if key is None:
            key = self._data.index[0]
        return self._data.counts[key]

    def process_metrics(self, metric_ftns, prefix, y_hat, y_true):
        # for met in itertools.chain.from_iterable(metric_ftns):
        for met in metric_ftns:
            report = met(y_hat, y_true)
            if isinstance(report, (list, tuple)):
                for r in report:
                    _n, _v = r
                    self.update(f'{prefix}{_n}', _v)
            else:
                self.update(f'{prefix}{met.__name__}', report)


def save_results_to_file(output_file: Path, log_dict: Dict, desc_dict: Dict = None):
    """
    Save results to a file
    :param output_file: output file
    :param log_dict: dictionary of key-value pairs
    :param desc_dict: optional dictionary of key-value pairs
    """
    output_file = Path(output_file)
    columns_metrics = sorted([k for k in log_dict])
    if desc_dict is not None:
        log_dict.update(desc_dict)

    df_log = pd.DataFrame({k: [v] for k, v in log_dict.items()})
    if desc_dict is not None:
        columns = list(desc_dict.keys()) + columns_metrics
        assert len(columns) == len(set(columns)), 'column names must be unique'
        df_log = df_log[columns]

    if output_file.exists():
        df = pd.read_excel(output_file)
        df_log = pd.concat([df, df_log], axis=0, join='outer', ignore_index=True)

    df_log.to_excel(output_file, index=False)

    # min results
    datasets = df_log['DATASET'].unique()

    # Initialize an empty DataFrame to store the results
    results = []

    # Loop through each dataset to find the maximum "valid_A/metrics/macro_avg_f1-score"
    for dataset in datasets:
        col_name = f'valid_{dataset}/metrics/macro_avg_f1-score'
        if col_name in df_log.columns:
            max_f1_score = df_log.loc[df_log['DATASET'] == dataset, col_name].max()
            results.append((dataset, max_f1_score))

    results_df = pd.DataFrame(results, columns=['DATASET', 'max_macro_avg_f1-score'])

    results_df.to_excel(output_file.with_name("results_macro_f1.xlsx"), index=False)

