from os.path import join

import torch
from code2seq.utils.vocabulary import Vocabulary
from omegaconf import DictConfig
from pl_bolts.models.self_supervised import SimCLR
from torchmetrics.functional import auroc

from models.encoders import encoder_models
from models.self_supervised.utils import (
    validation_metrics,
    prepare_features,
    clone_classification_step,
    compute_num_samples,
    scale
)


class SimCLRModel(SimCLR):
    def __init__(
        self,
        config: DictConfig,
        **kwargs
    ):
        self.config = config
        self.base_encoder = config.name
        train_data_path = join(
            config.data_folder,
            config.dataset.name,
            "raw",
            config.train_holdout
        )

        num_samples = compute_num_samples(train_data_path)

        super().__init__(
            gpus=-1,
            num_samples=num_samples,
            batch_size=config.hyper_parameters.batch_size,
            dataset="",
            num_nodes=config.ssl.num_nodes,
            hidden_mlp=config.num_classes,
            feat_dim=config.num_classes,
            warmup_epochs=config.ssl.warmup_epochs,
            max_epochs=config.ssl.max_epochs,
            temperature=config.ssl.temperature,
            optimizer=config.ssl.optimizer,
            exclude_bn_bias=config.ssl.exclude_bn_bias,
            start_lr=config.ssl.start_lr,
            learning_rate=config.ssl.learning_rate,
            final_lr=config.ssl.final_lr,
            weight_decay=config.ssl.weight_decay,
            **kwargs
        )

    def init_model(self):
        if self.base_encoder == "transformer":
            encoder = encoder_models[self.base_encoder](self.config)
        elif self.base_encoder == "code2class":
            _vocab_path = join(
                self.config.data_folder,
                self.config.dataset.name,
                self.config.dataset.dir,
                self.config.vocabulary_name
            )
            _vocabulary = Vocabulary.load_vocabulary(_vocab_path)
            encoder = encoder_models[self.base_encoder](config=self.config, vocabulary=_vocabulary)
        elif self.config.name == "gnn":
            encoder = encoder_models[self.config.name](self.config)
        elif self.config.name == "code-transformer":
            encoder = encoder_models[self.config.name](self.config)
        else:
            raise ValueError(f"Unknown model: {self.config.name}")
        return encoder

    def forward(self, x):
        return self.encoder(x)

    def representation(self, q, k):
        # get h representations, bolts resnet returns a list
        h1 = self(q)
        h2 = self(k)

        # get z representations
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        return z1, z2

    def _loss(self, logits, mask):
        batch_size = mask.shape[0] // 2

        # compute logits
        anchor_dot_contrast = logits / self.temperature

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask, device=self.device),
            1,
            torch.arange(2 * batch_size, device=self.device).view(-1, 1),
            0
        )
        mask_ = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask_ * log_prob).sum(1) / mask_.sum(1)
        loss = -mean_log_prob_pos.view(2, batch_size).mean()

        return loss

    def training_step(self, batch, batch_idx):
        (q, k, _), labels = batch
        queries, keys = self.representation(q=q, k=k)
        features, labels = prepare_features(queries, keys, labels)
        logits, mask = clone_classification_step(features, labels)

        loss = self._loss(logits, mask)

        with torch.no_grad():
            logits = scale(logits)
            logits = logits.reshape(-1)
            mask = mask.reshape(-1)

            roc_auc = auroc(logits, mask)

        self.log_dict({"train_loss": loss, "train_roc_auc": roc_auc})
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        features = self(features)
        features = self.projection(features)
        labels = labels.contiguous().view(-1, 1)

        return {"features": features, "labels": labels}

    def validation_epoch_end(self, outputs):
        log = validation_metrics(outputs)
        self.log_dict(log)


class SimCLRTransform:
    def __call__(self, batch):
        (x1, x2), y = batch
        return (x1, x2, None), y
