from dataclasses import dataclass


@dataclass
class LSTMConfig:
    vocab_size: int
    hidden_size: int
    output_size: int
    embedding_size: int
    bias: bool
    batch_first: bool
    dropout: float
    bidirectional: bool


test_config = LSTMConfig(
    embedding_size=8,
    hidden_size=8,
    output_size=8,
    vocab_size=30000,
    bias=False,
    batch_first=True,
    dropout=0.5,
    bidirectional=False
)

default_config = LSTMConfig(
    embedding_size=64,
    hidden_size=64,
    output_size=128,
    vocab_size=30000,
    bias=False,
    batch_first=True,
    dropout=0.5,
    bidirectional=False
)
