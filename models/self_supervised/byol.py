from copy import deepcopy
from os.path import join

import torch
import torch.nn.functional as F
from code2seq.utils.vocabulary import Vocabulary
from omegaconf import DictConfig
from pl_bolts.models.self_supervised import BYOL
from torchmetrics.functional import auroc

from models.encoders import SiameseArm
from models.encoders import encoder_models
from models.self_supervised.utils import validation


class BYOLModel(BYOL):
    def __init__(
        self,
        config: DictConfig,
        **kwargs
    ):
        self.config = config

        super().__init__(
            num_classes=config.num_classes,
            learning_rate=self.config.ssl.learning_rate,
            weight_decay=self.config.ssl.weight_decay,
            input_height=self.config.ssl.input_height,
            batch_size=self.config.ssl.batch_size,
            num_workers=self.config.ssl.num_workers,
            warmup_epochs=self.config.ssl.warmup_epochs,
            max_epochs=self.config.ssl.max_epochs,
            **kwargs
        )

        self._init_encoders()

    def _init_encoders(self):
        if self.config.name == "transformer":
            encoder = encoder_models[self.config.name](self.config)
        elif self.config.name == "code2class":
            _vocab_path = join(
                self.config.data_folder,
                self.config.dataset.name,
                self.config.dataset.dir,
                self.config.vocabulary_name
            )
            _vocabulary = Vocabulary.load_vocabulary(_vocab_path)
            encoder = encoder_models[self.config.name](config=self.config, vocabulary=_vocabulary)
        elif self.config.name == "gnn":
            encoder = encoder_models[self.config.name](self.config)
        else:
            raise ValueError(f"Unknown model: {self.config.name}")
        self.online_network = SiameseArm(
            encoder=encoder,
            input_dim=self.config.num_classes,
            output_dim=self.config.num_classes,
            hidden_size=self.config.num_classes
        )
        self.target_network = deepcopy(self.online_network)

    def shared_step(self, batch, batch_idx):
        imgs, y = batch
        q, k = imgs[:2]

        # Image 1 to image 2 loss
        y1, z1, h1 = self.online_network(q)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(k)
        loss_a = -2 * F.cosine_similarity(h1, z2).mean()

        # Image 2 to image 1 loss
        y1, z1, h1 = self.online_network(k)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(q)
        # L2 normalize
        loss_b = -2 * F.cosine_similarity(h1, z2).mean()

        # Final loss
        total_loss = loss_a + loss_b

        return loss_a, loss_b, total_loss, h1, h2

    def training_step(self, batch, batch_idx):
        *_, labels = batch
        loss_a, loss_b, total_loss, h1, h2 = self.shared_step(batch, batch_idx)

        with torch.no_grad():
            features = torch.cat([h1, h2], dim=0)
            labels = labels.contiguous().view(-1, 1)
            labels = labels.repeat(2, 1)

            logits = torch.matmul(features, features.T).reshape(-1)
            logits = (logits - logits.min()) / (logits.max() - logits.min())
            mask = torch.eq(labels, labels.T).reshape(-1)

            roc_auc = auroc(logits, mask)

        log = {"train_loss": total_loss, "train_roc_auc": roc_auc}
        self.log_dict(log)
        return total_loss

    def validation_step(self, batch, batch_idx):
        *_, labels = batch
        loss_a, loss_b, total_loss, h1, h2 = self.shared_step(batch, batch_idx)
        features = torch.cat([h1, h2], dim=0)
        labels = labels.contiguous().view(-1, 1)
        labels = labels.repeat(2, 1)

        log = {"loss": total_loss, "features": features, "labels": labels}
        return log

    def validation_epoch_end(self, outputs):
        log = validation(outputs)
        self.log_dict(log)


class BYOLTransform:
    def __call__(self, batch):
        (x1, x2), y = batch
        return (x1, x2, None), y
