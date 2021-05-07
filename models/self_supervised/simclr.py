from os import listdir
from os.path import join, isdir

import torch
from code2seq.utils.vocabulary import Vocabulary
from omegaconf import DictConfig
from pl_bolts.metrics import mean, precision_at_k
from pl_bolts.models.self_supervised import SimCLR

from models.encoders import encoder_models


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
                num_samples += len([_ for _ in listdir(class_path)])

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

    def shared_step(self, batch):
        # final image in tuple is for online eval
        (img1, img2, _), y = batch

        # get h representations, bolts resnet returns a list
        h1 = self(img1)
        h2 = self(img2)

        # get z representations
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2, self.temperature)

        with torch.no_grad():
            logits = torch.mm(z1, z2.T)
            labels = torch.arange(logits.shape[0], dtype=torch.long)
            labels = labels.type_as(logits)
            acc1, acc5 = precision_at_k(logits, labels, top_k=(1, 5))

        return loss, acc1, acc5

    def training_step(self, batch, batch_idx):
        loss, acc1, acc5 = self.shared_step(batch)

        log = {'train_loss': loss, 'train_acc1': acc1, 'train_acc5': acc5}
        self.log_dict(log)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc1, acc5 = self.shared_step(batch)

        logs = {'val_loss': loss, 'val_acc1': acc1, 'val_acc5': acc5}
        return logs

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, 'val_loss')
        val_acc1 = mean(outputs, 'val_acc1')
        val_acc5 = mean(outputs, 'val_acc5')

        log = {'val_loss': val_loss, 'val_acc1': val_acc1, 'val_acc5': val_acc5}
        self.log_dict(log)


class SimCLRTransform:
    def __call__(self, batch):
        (x1, x2), y = batch
        return (x1, x2, None), y
