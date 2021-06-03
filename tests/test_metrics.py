import torch
from torchmetrics.functional import confusion_matrix

import numpy
from models.self_supervised.utils import compute_f1, compute_map_at_k
from sklearn.metrics import f1_score


def test_f1():
    preds = [0, 1, 0, 1, 1, 1, 1, 0, 1]
    target = [0, 1, 1, 0, 1, 1, 1, 0, 0]
    sklearn_f1 = f1_score(target, preds)
    conf_matrix = confusion_matrix(torch.LongTensor(preds), torch.LongTensor(target), num_classes=2)
    custom_f1 = compute_f1(conf_matrix)
    assert numpy.isclose(custom_f1, sklearn_f1)


def _test(preds, result):
    map_at_k = compute_map_at_k(preds)
    assert numpy.isclose(map_at_k, result)


def test_map_at_k():
    preds = torch.Tensor(
        [
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False]
        ]
    )
    _test(preds, 0)
    preds = torch.Tensor(
        [
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
        ]
    )
    _test(preds, (1 + 1 / 2 + 1 / 3 + 1 / 4) / 16)
    preds = torch.Tensor(
        [
            [True, False, True, False],
            [False, False, True, True],
            [True, True, True, True],
            [False, False, False, False],
        ]
    )
    _test(preds, ((1 + 2 / 3) + (1 / 3 + 2 / 4) + 4) / 16)
    preds = torch.Tensor(
        [
            [True, True, True, False],
            [False, False, True, True],
            [True, True, False, True],
        ]
    )
    _test(preds, (3 + (1 / 3 + 2 / 4) + (2 + 3 / 4)) / 12)
    preds = torch.Tensor(
        [
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
            [True, False, True, False],
            [False, False, True, True],
            [True, True, True, True],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False]
        ]
    )
    _test(preds, (1 + 1 / 2 + 1 / 3 + 1 / 4 + (1 + 2 / 3) + (1 / 3 + 2 / 4) + 4 ) / 40)
