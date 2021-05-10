from os.path import join

import torch
from code2seq.utils.vocabulary import Vocabulary
from omegaconf import DictConfig
from pl_bolts.metrics import precision_at_k
from pl_bolts.models.self_supervised import Moco_v2
from pl_bolts.models.self_supervised.moco.moco2_module import concat_all_gather
from torch import nn
from torch.nn import functional as F

from models.encoders import encoder_models


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
        self.register_buffer("val_queue", torch.randn(config.num_classes, config.ssl.num_negatives))
        self.queue = F.normalize(self.val_queue, dim=0)

        self.register_buffer("val_queue_ptr", torch.zeros(1, dtype=torch.long))

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

    def forward(self, img_q, img_k, queue):
        # compute query features
        q = self.encoder_q(img_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys

            # shuffle for making use of BN
            if self.trainer.use_ddp or self.trainer.use_ddp2:
                img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            k = self.encoder_k(img_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            if self.trainer.use_ddp or self.trainer.use_ddp2:
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.hparams.softmax_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        return logits, labels, k

    def compute_metrcis(self, output, target):
        loss = F.cross_entropy(output.float(), target.long())
        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))
        return loss, acc1, acc5

    def training_step(self, batch, batch_idx):
        (img_1, img_2), _ = batch

        self._momentum_update_key_encoder()  # update the key encoder
        output, target, keys = self(img_q=img_1, img_k=img_2, queue=self.queue)
        self._dequeue_and_enqueue(keys, queue=self.queue, queue_ptr=self.queue_ptr)  # dequeue and enqueue

        loss, acc1, acc5 = self.compute_metrcis(output, target)

        log = {'train_loss': loss, 'train_acc1': acc1, 'train_acc5': acc5}
        self.log_dict(log)
        return loss

    def validation_step(self, batch, batch_idx):
        (img_1, img_2), labels = batch

        output, target, keys = self(img_q=img_1, img_k=img_2, queue=self.val_queue)
        self._dequeue_and_enqueue(keys, queue=self.val_queue, queue_ptr=self.val_queue_ptr)  # dequeue and enqueue

        loss, acc1, acc5 = self.compute_metrcis(output, target)

        results = {'val_loss': loss, 'val_acc1': acc1, 'val_acc5': acc5}
        return results
