import torch
from torch_cluster import knn


def compute_f1(conf_matrix):
    assert conf_matrix.shape == (2, 2)
    tn, fn, fp, tp = conf_matrix.reshape(-1).tolist()
    f1 = tp / (tp + 0.5 * (fp + fn))
    return f1


def compute_map_at_k(preds):
    avg_precisions = []
    for pred in preds:
        positions = torch.arange(1, pred.shape[0] + 1, device=preds.device)[pred > 0]
        if positions.shape[0]:
            avg = torch.arange(1, positions.shape[0] + 1, device=positions.device) / positions
            avg_precisions.append(avg.mean())
    if avg_precisions:
        return torch.stack(avg_precisions).mean().item()
    else:
        return 0


def validation_metrics(outputs):
    features = torch.cat([out["features"] for out in outputs])
    _, hidden_size = features.shape

    labels = torch.cat([out["labels"] for out in outputs]).reshape(-1)

    logs = {}
    for k in [5, 10, 15]:
        if k < labels.shape[0]:
            top_ids = knn(x=features, y=features, k=k + 1)
            top_ids = top_ids[1, :].reshape(-1, k + 1)
            top_ids = top_ids[:, 1:]

            top_labels = labels[top_ids]
            preds = torch.eq(top_labels, labels.reshape(-1, 1))
            logs[f"val_map@{k}"] = compute_map_at_k(preds)
    return logs


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
