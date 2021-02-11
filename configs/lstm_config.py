from dataclasses import dataclass

from .tokenizer_config import TokenizerConfig


@dataclass
class LSTMConfig:
    vocab_size: int
    hidden_size: int
    output_size: int
    embedding_size: int
    batch_first: bool
    bidirectional: bool
    dropout: float

    pad_id: int = TokenizerConfig.pad_id
    unk_id: int = TokenizerConfig.unk_id
    bos_id: int = TokenizerConfig.bos_id
    eos_id: int = TokenizerConfig.eos_id


test_config = LSTMConfig(
    embedding_size=8,
    hidden_size=8,
    output_size=8,
    vocab_size=200,
    batch_first=True,
    dropout=0.2,
    bidirectional=False
)

default_config = LSTMConfig(
    embedding_size=64,
    hidden_size=64,
    output_size=128,
    vocab_size=30000,
    batch_first=True,
    dropout=0.2,
    bidirectional=False
)
