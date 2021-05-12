import torch
from pl_bolts.metrics import mean
from torchmetrics.functional import auroc


def validation(outputs):
    val_loss = mean(outputs, "loss")
    features = torch.cat([out["features"] for out in outputs])
    labels = torch.cat([out["labels"] for out in outputs])

    logits = torch.mm(features, features.T)
    logits = (logits - logits.min()) / (logits.max() - logits.min())
    logits = logits.reshape(-1).half()

    mask = torch.eq(labels, labels.T)
    mask = mask.reshape(-1)

    val_roc_auc = auroc(logits, mask)
    return {"val_loss": val_loss, "val_roc_auc": val_roc_auc}
