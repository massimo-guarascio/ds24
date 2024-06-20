import torch
import warnings
import numpy as np

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report as classification_report_sklearn

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)


def accuracy(output, target):
    with torch.no_grad():
        # pred = torch.argmax(output, dim=1)
        pred = output.squeeze() >= .5
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def classification_report(output, target):
    with torch.no_grad():
        if isinstance(output, torch.Tensor):
            output = (output.squeeze() >= .5).cpu().numpy()
            target = target.cpu().numpy()

        report = classification_report_sklearn(target,
                                               output,
                                               target_names=['real', 'fake'],
                                               output_dict=True)
        r = []
        for k in report:
            if k == 'accuracy':
                r.append((f'{str(k)}', report[k]))
            else:
                for kk in report[k]:
                    if kk != 'support':
                        r.append((f'{str(k).replace(" ", "_")}_{kk}', report[k][kk]))
        return r


def get_label_classification_report():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 0, 1, 1, 1]

    report = classification_report_sklearn(y_true, y_pred,
                                           target_names=['real', 'fake'],
                                           output_dict=True)
    r = ['accuracy']
    for k in report:
        if k != 'accuracy':
            for kk in report[k]:
                if kk != 'support':
                    r.append(f'{str(k).replace(" ", "_")}_{kk}')

    return r
