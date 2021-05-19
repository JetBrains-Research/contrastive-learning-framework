import torch
from pl_bolts.metrics import mean
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.functional import confusion_matrix


def compute_f1(conf_matrix):
    assert conf_matrix.shape == (2, 2)
    tn, fn, fp, tp = conf_matrix.reshape(-1).tolist()
    f1 = tp / (tp + 0.5 * (fp + fn))
    return f1


def validation_metrics(outputs):
    features_all = torch.cat([out["features"] for out in outputs])
    labels_all = torch.cat([out["labels"] for out in outputs])

    dataset = TensorDataset(features_all, labels_all)

    a_loader = DataLoader(dataset, batch_size=1024)
    b_loader = DataLoader(dataset, batch_size=1024)

    device = features_all.device

    conf_matrix = torch.zeros((2, 2), device=device)

    for a_features, a_labels in a_loader:
        for b_features, b_labels in b_loader:
            logits = torch.matmul(a_features, b_features.T)
            mask = torch.eq(a_labels, b_labels.T)

            logits = logits.reshape(-1)
            logits = scale(logits)

            preds = (logits >= 0.5).long()
            mask = mask.reshape(-1)

            conf_matrix += confusion_matrix(preds, mask, num_classes=2)

    f1 = compute_f1(conf_matrix=conf_matrix)
    return {"val_f1": f1}


def clone_classification_step(features, labels):
    logits = torch.matmul(features, features.T)
    mask = torch.eq(labels, labels.T)
    return logits, mask


def prepare_features(queries, keys, labels):
    features = torch.cat([queries, keys], dim=0)
    labels = labels.contiguous().view(-1, 1)
    labels = labels.repeat(2, 1)
    return features, labels


def scale(x):
    x = torch.clamp(x, min=-1, max=1)
    return (x + 1) / 2
