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
    val_loss = mean(outputs, "loss")
    features_all = torch.cat([out["features"] for out in outputs])
    labels_all = torch.cat([out["labels"] for out in outputs])

    dataset = TensorDataset(features_all, labels_all)

    a_loader = DataLoader(dataset, batch_size=32)
    b_loader = DataLoader(dataset, batch_size=32)

    conf_matrix = torch.zeros((2, 2))

    for a_features, a_labels in a_loader:
        for b_features, b_labels in b_loader:
            features = torch.cat([a_features, b_features], dim=0)
            labels = torch.cat([a_labels, b_labels], dim=0)

            logits, mask = clone_classification_step(features, labels)
            logits = logits.reshape(-1)
            logits = scale(logits)

            preds = (logits >= 0.5).long()
            mask = mask.reshape(-1)

            conf_matrix += confusion_matrix(preds, mask, num_classes=2)

    f1 = compute_f1(conf_matrix=conf_matrix)
    return {"val_loss": val_loss, "val_f1": f1}


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
