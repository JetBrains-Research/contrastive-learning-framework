from copy import deepcopy
from os.path import join

import torch.nn as nn
from code2seq.utils.vocabulary import Vocabulary
from omegaconf import DictConfig
from pl_bolts.models.self_supervised import BYOL
from pl_bolts.models.self_supervised.byol.models import MLP

from models.encoders import encoder_models


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
        if self.config.name == "lstm":
            encoder = encoder_models[self.config.name](self.config)
        elif self.config.name == "code2class":
            _vocab_path = join(self.config.data_folder, self.config.dataset.name, self.config.vocabulary_name)
            _vocabulary = Vocabulary.load_vocabulary(_vocab_path)
            encoder = encoder_models[self.config.name](config=self.config, vocabulary=_vocabulary)
        else:
            print(f"Unknown model: {self.config.name}")
        self.online_network = SiameseArm(encoder=encoder)
        self.target_network = deepcopy(self.online_network)


class BYOLTransform:
    def __call__(self, batch):
        (x1, x2), y = batch
        return (x1, x2, None), y


class SiameseArm(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()
        # Encoder
        self.encoder = encoder
        # Projector
        self.projector = MLP(
            input_dim=encoder.num_classes,
            hidden_size=128,
            output_dim=128
        )
        # Predictor
        self.predictor = MLP(
            input_dim=128,
            hidden_size=128,
            output_dim=128
        )

    def forward(self, x):
        y = self.encoder(x)
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h
