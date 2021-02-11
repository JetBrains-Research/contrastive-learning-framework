from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    vocab_size: int
    n_threads: int
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3


test_tokenizer_config = TokenizerConfig(
    vocab_size=200,
    n_threads=-1
)

default_tokenizer_config = TokenizerConfig(
    vocab_size=30000,
    n_threads=-1
)
