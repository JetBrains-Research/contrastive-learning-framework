import torch
import torch.nn as nn
from math import log
from omegaconf import DictConfig


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()

        pos_embedding = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1)

        # PE(position, 2i)     = sin( position / 10000 ** (2i / d_model) )
        # PE(position, 2i + 1) = cos( position / 10000 ** (2i / d_model) )
        div_term = torch.exp(log(10000.0) * torch.arange(0, d_model, 2) / d_model)

        pos_embedding[:, 0::2] = torch.sin(position / div_term)
        pos_embedding[:, 1::2] = torch.cos(position / div_term)

        self.pos_embedding = nn.Parameter(
            pos_embedding.unsqueeze(0),
            requires_grad=False
        )

    def forward(self, x):
        return self.pos_embedding[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.num_classes = config.num_classes
        self.hidden_size = config.encoder.hidden_size

        self.token_embedding = nn.Embedding(
            config.dataset.vocab_size,
            config.encoder.hidden_size,
            padding_idx=config.dataset.pad_id,
        )
        self.position_embedding = PositionalEmbedding(
            d_model=config.encoder.hidden_size,
            max_seq_len=config.encoder.max_seq_len
        )
        self.dropout = nn.Dropout()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder.hidden_size,
            nhead=config.encoder.num_heads
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder.num_layers
        )
        self.fc = nn.Linear(config.encoder.hidden_size, config.num_classes)

    def forward(self, seq: torch.Tensor):
        out = self.dropout(self.token_embedding(seq) + self.position_embedding(seq))
        out = self.transformer(out)
        return self.fc(out.mean(0))
