from dataclasses import dataclass, asdict

from pl_bolts.models.self_supervised import MocoV2

from models.encoders import encoder_models


class MocoV2Model(MocoV2):
    def __init__(
        self,
        base_encoder: str,
        encoder_config: dataclass,
        num_negatives: int = 65536,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        use_mlp: bool = False,
        num_workers: int = 8,
        **kwargs
    ):
        self.hparams = asdict(encoder_config)
        self.encoder_config = encoder_config
        super().__init__(
            base_encoder=base_encoder,
            emb_dim=encoder_config.output_size,
            num_negatives=num_negatives,
            encoder_momentum=encoder_momentum,
            softmax_temperature=softmax_temperature,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            batch_size=batch_size,
            use_mlp=use_mlp,
            num_workers=num_workers,
            **kwargs
        )

    def init_encoders(self, base_encoder: str):
        encoder_q = encoder_models[base_encoder](self.encoder_config)
        encoder_k = encoder_models[base_encoder](self.encoder_config)
        return encoder_q, encoder_k
