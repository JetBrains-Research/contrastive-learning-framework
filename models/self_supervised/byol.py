from copy import deepcopy
from os.path import join

import torch
import torch.nn.functional as F
from code2seq.data.vocabulary import Vocabulary
from omegaconf import DictConfig
from pl_bolts.models.self_supervised import BYOL
from torchmetrics.functional import auroc

from models.encoders import SiameseArm
from models.encoders import encoder_models
from models.self_supervised.utils import validation_metrics, prepare_features, clone_classification_step, scale


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
            _vocabulary = Vocabulary(
                join(
                    self.config.data_folder,
                    self.config.dataset.name,
                    self.config.dataset.dir,
                    self.config.vocabulary_name
                ),
                self.config.dataset.max_labels,
                self.config.dataset.max_tokens
            )
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

    def _loss(self, h1_1, h2_1, z1_2, z2_2):
        loss_a = -2 * F.cosine_similarity(h1_1, z1_2).mean()
        loss_b = -2 * F.cosine_similarity(h2_1, z2_2).mean()
        return loss_a + loss_b

    def representation(self, q, k):
        # Image 1 to image 2 loss
        y1_1, z1_1, h1_1 = self.online_network(q)
        with torch.no_grad():
            y1_2, z1_2, h1_2 = self.target_network(k)

        # Image 2 to image 1 loss
        y2_1, z2_1, h2_1 = self.online_network(k)
        with torch.no_grad():
            y2_2, z2_2, h2_2 = self.target_network(q)

        return h1_1, h2_1, z1_2, z2_2

    def training_step(self, batch, batch_idx):
        (q, k, _), labels = batch

        h1_1, h2_1, z1_2, z2_2 = self.representation(q=q, k=k)
        loss = self._loss(h1_1, h2_1, z1_2, z2_2)
        queries, keys = h1_1, h2_1

        with torch.no_grad():
            features, labels = prepare_features(queries, keys, labels)
            logits, mask = clone_classification_step(features, labels)
            logits = scale(logits)
            logits = logits.reshape(-1)
            mask = mask.reshape(-1)

            roc_auc = auroc(logits, mask)

        self.log_dict({"train_loss": loss, "train_roc_auc": roc_auc})
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        *_, features = self.online_network(features)
        features = F.normalize(features, dim=1)
        labels = labels.contiguous().view(-1, 1)

        return {"features": features, "labels": labels}

    def validation_epoch_end(self, outputs):
        log = validation_metrics(outputs)
        self.log_dict(log)


class BYOLTransform:
    def __call__(self, batch):
        (x1, x2), y = batch
        return (x1, x2, None), y
