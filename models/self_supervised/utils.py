import torch
from pl_bolts.metrics import mean
from torchmetrics.functional import auroc


def validation_metrics(outputs):
    val_loss = mean(outputs, "loss")
    features = torch.cat([out["features"] for out in outputs])
    labels = torch.cat([out["labels"] for out in outputs])

    logits, mask = clone_classification_step(features, labels)
    logits = min_max_scale(logits)

    val_roc_auc = auroc(logits.reshape(-1), mask.reshape(-1))
    return {"val_loss": val_loss, "val_roc_auc": val_roc_auc}


def clone_classification_step(features, labels):
    logits = torch.matmul(features, features.T)
    mask = torch.eq(labels, labels.T)
    return logits, mask


def prepare_features(queries, keys, labels):
    features = torch.cat([queries, keys], dim=0)
    labels = labels.contiguous().view(-1, 1)
    labels = labels.repeat(2, 1)
    return features, labels


def min_max_scale(x):
    return (x - x.min()) / (x.max() - x.min())
