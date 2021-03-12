from copy import deepcopy
from dataclasses import dataclass, asdict

import torch.nn as nn
from code2seq.utils.vocabulary import Vocabulary
from omegaconf import OmegaConf
from pl_bolts.models.self_supervised import BYOL
from pl_bolts.models.self_supervised.byol.models import MLP

from models.encoders import encoder_models


class BYOLModel(BYOL):
    def __init__(
        self,
        base_encoder: str,
        encoder_config: dataclass,
        learning_rate: float = 0.2,
        weight_decay: float = 1.5e-6,
        input_height: int = 32,
        batch_size: int = 32,
        num_workers: int = -1,
        warmup_epochs: int = 10,
        max_epochs: int = 1000,
        **kwargs
    ):
        self.hparams = asdict(encoder_config)
        self.encoder_config = encoder_config

        if base_encoder == "LSTM":
            encoder = encoder_models[base_encoder](self.encoder_config)
            self.num_classes = self.encoder_config.num_classes
        else:
            _config = OmegaConf.load("configs/code2class-poj104.yaml")
            _vocabulary = Vocabulary.load_vocabulary("data/poj_104/vocabulary.pkl")
            encoder = encoder_models[base_encoder](config=_config, vocabulary=_vocabulary)
            self.num_classes = _config.num_classes

        super().__init__(
            num_classes=self.num_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            input_height=input_height,
            batch_size=batch_size,
            num_workers=num_workers,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            **kwargs
        )

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