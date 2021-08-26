from os import listdir
from os.path import isdir, join

import torch
from code2seq.data.vocabulary import Vocabulary
from torch_cluster import knn
from torch.optim import Adam
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics.functional import auroc

from models import encoder_models


def configure_optimizers(params, learning_rate, weight_decay, warmup_epochs, max_epochs):
    optimizer = Adam(params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, max_epochs=max_epochs)
    return [optimizer], [scheduler]


def init_model(config):
    if config.name == "transformer":
        encoder = encoder_models[config.name](config)
    elif config.name == "code2class":
        _vocabulary = Vocabulary(
            join(
                config.data_folder,
                config.dataset.name,
                config.dataset.dir,
                config.vocabulary_name
            ),
            config.dataset.max_labels,
            config.dataset.max_tokens
        )
        encoder = encoder_models[config.name](config=config, vocabulary=_vocabulary)
    elif config.name == "gnn":
        encoder = encoder_models[config.name](config)
    else:
        raise ValueError(f"Unknown model: {config.name}")
    return encoder


@torch.no_grad()
def roc_auc(queries, keys, labels):
    features, labels = prepare_features(queries, keys, labels)
    logits, mask = clone_classification_step(features, labels)
    logits = scale(logits)
    logits = logits.reshape(-1)
    mask = mask.reshape(-1)

    return auroc(logits, mask)


def compute_f1(conf_matrix):
    assert conf_matrix.shape == (2, 2)
    tn, fn, fp, tp = conf_matrix.reshape(-1).tolist()
    f1 = tp / (tp + 0.5 * (fp + fn))
    return f1


def compute_map_at_k(preds):
    avg_precisions = []

    k = preds.shape[1]
    for pred in preds:
        positions = torch.arange(1, k + 1, device=preds.device)[pred > 0]
        if positions.shape[0]:
            avg = torch.arange(1, positions.shape[0] + 1, device=positions.device) / positions
            avg_precisions.append(avg.sum() / k)
        else:
            avg_precisions.append(torch.tensor(0.0, device=preds.device))
    return torch.stack(avg_precisions).mean().item()


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


def compute_num_samples(train_data_path: str):
    num_samples = 0
    for class_ in listdir(train_data_path):
        class_path = join(train_data_path, class_)
        if isdir(class_path):
            num_files = len([_ for _ in listdir(class_path)])
            num_samples += num_files * (num_files - 1) // 2
    return num_samples
