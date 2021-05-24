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


def test_map_at_k():
    preds = torch.Tensor(
        [
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
            [True, False, True, False],
            [True, True, True, True],
        ]
    )
    map_at_k = compute_map_at_k(preds)
    assert numpy.isclose(map_at_k * 6, (1 + 1/2 + 1/3 + 1/4 + 0.5 * (1 + 2/3) + 1))
