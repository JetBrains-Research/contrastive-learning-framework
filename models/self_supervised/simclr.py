from os import listdir
from os.path import join, isdir

import torch
from code2seq.utils.vocabulary import Vocabulary
from omegaconf import DictConfig
from pl_bolts.models.self_supervised import SimCLR
from torchmetrics.functional import auroc

from models.encoders import encoder_models
from .utils import validation_metrics, prepare_features, min_max_scale, clone_classification_step


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
            config.dataset.dir,
            config.train_holdout
        )

        num_samples = 0
        for class_ in listdir(train_data_path):
            class_path = join(train_data_path, class_)
            if isdir(class_path):
                num_files = len([_ for _ in listdir(class_path)])
                num_samples += num_files * (num_files - 1) // 2
        super().__init__(
            gpus=-1,
            num_samples=num_samples,
            batch_size=config.hyper_parameters.batch_size,
            dataset="",
            num_nodes=config.ssl.num_nodes,
            hidden_mlp=config.encoder.num_classes,
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
            torch.ones_like(mask),
            1,
            torch.arange(2 * batch_size).view(-1, 1),
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
            logits = min_max_scale(logits)
            roc_auc = auroc(logits.reshape(-1), mask.reshape(-1))

        self.log_dict({"train_loss": loss, "train_roc_auc": roc_auc})
        return loss

    def validation_step(self, batch, batch_idx):
        (q, k, _), labels = batch
        queries, keys = self.representation(q=q, k=k)
        features, labels = prepare_features(queries, keys, labels)
        logits, mask = clone_classification_step(features, labels)

        loss = self._loss(logits, mask)

        return {"loss": loss, "features": features, "labels": labels}

    def validation_epoch_end(self, outputs):
        log = validation_metrics(outputs)
        self.log_dict(log)


class SimCLRTransform:
    def __call__(self, batch):
        (x1, x2), y = batch
        return (x1, x2, None), y
