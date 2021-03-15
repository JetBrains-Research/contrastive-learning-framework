from dataclasses import dataclass

from .tokenizer_config import TokenizerConfig


@dataclass
class LSTMConfig:
    vocab_size: int
    hidden_size: int
    num_classes: int
    embedding_size: int
    bidirectional: bool
    dropout: float

    pad_id: int = TokenizerConfig.pad_id
    unk_id: int = TokenizerConfig.unk_id
    bos_id: int = TokenizerConfig.bos_id
    eos_id: int = TokenizerConfig.eos_id


test_config = LSTMConfig(
    embedding_size=8,
    hidden_size=8,
    num_classes=5,
    vocab_size=10000,
    dropout=0.2,
    bidirectional=False
)

default_config = LSTMConfig(
    embedding_size=64,
    hidden_size=64,
    num_classes=128,
    vocab_size=10000,
    dropout=0.2,
    bidirectional=False
)
