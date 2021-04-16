import torch
import torch.nn as nn
from omegaconf import DictConfig


class TransformerModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.num_classes = config.num_classes
        self.hidden_size = config.encoder.hidden_size
        self.embeddings = nn.Embedding(
            config.dataset.vocab_size,
            config.encoder.hidden_size,
            padding_idx=config.dataset.pad_id,
        )
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
        out = self.embeddings(seq)
        out = self.transformer(out)
        return self.fc(out.mean(0))
