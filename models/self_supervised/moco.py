from os.path import join

import torch
from code2seq.utils.vocabulary import Vocabulary
from omegaconf import DictConfig
from pl_bolts.models.self_supervised import Moco_v2
from pl_bolts.models.self_supervised.moco.moco2_module import concat_all_gather
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional import auroc

from models.encoders import encoder_models
from .utils import validation_metrics, prepare_features, clone_classification_step, scale


class MocoV2Model(Moco_v2):
    def __init__(
        self,
        config: DictConfig,
        **kwargs
    ):
        self.config = config

        super().__init__(
            base_encoder=config.name,
            emb_dim=config.num_classes,
            num_negatives=config.ssl.num_negatives,
            encoder_momentum=config.ssl.encoder_momentum,
            softmax_temperature=config.ssl.softmax_temperature,
            learning_rate=config.ssl.learning_rate,
            momentum=config.ssl.momentum,
            weight_decay=config.ssl.weight_decay,
            batch_size=config.ssl.batch_size,
            use_mlp=config.ssl.use_mlp,
            num_workers=config.ssl.num_workers,
            **kwargs
        )

        # create the validation queue
        self.register_buffer("labels_queue", torch.zeros(1, config.ssl.num_negatives))
        self.labels_queue = F.normalize(self.labels_queue, dim=0)

        self.register_buffer("labels_queue_ptr", torch.zeros(1, dtype=torch.long))

    def init_encoders(self, base_encoder: str):
        if base_encoder == "transformer":
            encoder_q = encoder_models[base_encoder](self.config)
            encoder_k = encoder_models[base_encoder](self.config)
        elif base_encoder == "code2class":
            _vocab_path = join(
                self.config.data_folder,
                self.config.dataset.name,
                self.config.dataset.dir,
                self.config.vocabulary_name
            )
            _vocabulary = Vocabulary.load_vocabulary(_vocab_path)
            encoder_q = encoder_models[base_encoder](config=self.config, vocabulary=_vocabulary)
            encoder_k = encoder_models[base_encoder](config=self.config, vocabulary=_vocabulary)
        elif self.config.name == "gnn":
            encoder_q = encoder_models[self.config.name](self.config)
            encoder_k = encoder_models[self.config.name](self.config)
        else:
            raise ValueError(f"Unknown model: {self.config.name}")
        return encoder_q, encoder_k

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue_ptr, queue):
        # gather keys before updating queue
        if self.trainer.use_ddp or self.trainer.use_ddp2:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(queue_ptr)
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        queue_ptr[0] = ptr

    def forward(self, input_):
        q = self.encoder_q(input_)
        q = nn.functional.normalize(q, dim=1)
        return q

    def representation(self, q, k):
        # compute query features
        q = self.encoder_q(q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys

            # shuffle for making use of BN
            if self.trainer.use_ddp or self.trainer.use_ddp2:
                img_k, idx_unshuffle = self._batch_shuffle_ddp(k)

            k = self.encoder_k(k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            if self.trainer.use_ddp or self.trainer.use_ddp2:
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        return q, k

    def uni_con(self, logits, target):
        sum_neg = ((1 - target) * torch.exp(logits)).sum(1)
        sum_pos = (target * torch.exp(-logits)).sum(1)
        loss = torch.log(1 + sum_neg * sum_pos)
        return torch.mean(loss)

    def _loss(self, q, k, labels, queue, labels_queue):
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.hparams.softmax_temperature

        # positive label for the augmented version
        target_aug = torch.ones((q.shape[0], 1), device=q.device)
        # comparing the query label with l_que
        target_que = torch.eq(labels.reshape(-1, 1), labels_queue.reshape(1, -1))
        # labels: Nx(1+K)
        target = torch.cat([target_aug, target_que], dim=1)
        # calculate the contrastive loss, Eqn.(7)
        loss = self.uni_con(logits, target)
        return loss

    def training_step(self, batch, batch_idx):
        (q, k), labels = batch

        # update the key encoder
        self._momentum_update_key_encoder()

        queries, keys = self.representation(q=q, k=k)
        loss = self._loss(
            q=queries,
            k=keys,
            labels=labels,
            queue=self.queue,
            labels_queue=self.labels_queue
        )

        # dequeue and enqueue
        self._dequeue_and_enqueue(keys, queue=self.queue, queue_ptr=self.queue_ptr)
        self._dequeue_and_enqueue(labels, queue=self.labels_queue, queue_ptr=self.labels_queue_ptr)

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
        features = self(features)
        labels = labels.contiguous().view(-1, 1)

        return {"features": features, "labels": labels}

    def validation_epoch_end(self, outputs):
        log = validation_metrics(outputs)
        self.log_dict(log)
