from os.path import join

from code2seq.utils.vocabulary import Vocabulary
from omegaconf import DictConfig
from pl_bolts.models.self_supervised import Moco_v2

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
