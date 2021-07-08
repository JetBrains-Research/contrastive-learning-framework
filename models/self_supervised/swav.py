from os import listdir
from os.path import join, isdir

import torch
import numpy as np
from code2seq.utils.vocabulary import Vocabulary
from omegaconf import DictConfig
from pl_bolts.models.self_supervised import SwAV
from pl_bolts.models.self_supervised.swav.swav_resnet import MultiPrototypes
from torchmetrics.functional import auroc

from models.encoders import encoder_models
from models.self_supervised.utils import (
    validation_metrics,
    prepare_features,
    clone_classification_step,
    compute_num_samples,
    scale
)


class SwAVModel(SwAV):
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
            gpus=config.ssl.gpus,
            num_samples=num_samples,
            batch_size=config.hyper_parameters.batch_size,
            dataset="",
            nmb_crops=config.ssl.nmb_crops,
        )

        self.prototypes = None
        if isinstance(config.ssl.nmb_prototypes, list):
            self.prototypes = MultiPrototypes(config.ssl.feat_dim, config.ssl.nmb_prototypes)
        elif config.ssl.nmb_prototypes > 0:
            self.prototypes = torch.nn.Linear(config.ssl.feat_dim, config.ssl.nmb_prototypes, bias=False)

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
        return self.model(x)

    def _loss(self, embedding, output, bs):
        # 3. swav loss computation
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id:bs * (crop_id + 1)]

                # 4. time to use the queue
                if self.queue is not None:
                    if self.use_the_queue or not torch.all(self.queue[i, -1, :] == 0):
                        self.use_the_queue = True
                        out = torch.cat((torch.mm(self.queue[i], self.prototypes.weight.t()), out))
                    # fill the queue
                    self.queue[i, bs:] = self.queue[i, :-bs].clone()
                    self.queue[i, :bs] = embedding[crop_id * bs:(crop_id + 1) * bs]

                # 5. get assignments
                q = torch.exp(out / self.epsilon).t()
                q = self.get_assignments(q, self.sinkhorn_iterations)[-bs:]

            # cluster assignment prediction
            subloss = 0

            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                p = self.softmax(output[bs * v:bs * (v + 1)] / self.temperature)
                subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
            loss += subloss / (np.sum(self.nmb_crops) - 1)
        loss /= len(self.crops_for_assign)

        return loss

    def training_step(self, batch, batch_idx):
        (q, k), labels = batch

        # 1. normalize the prototypes
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = torch.nn.functional.normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)

        # 2. multi-res forward passes
        embedding = torch.cat((self.model(q), self.model(k)))
        output = self.prototypes(embedding)
        embedding = embedding.detach()

        loss = self._loss(embedding, output, bs=labels.shape[0])

        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        features = self(inputs)
        labels = labels.contiguous().view(-1, 1)

        return {"features": features, "labels": labels}

    def validation_epoch_end(self, outputs):
        log = validation_metrics(outputs)
        self.log_dict(log)
