import torch
import torch.nn as nn
from omegaconf import DictConfig


class TransformerModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.num_classes = config.num_classes
        self.hidden_size = config.encoder.hidden_size
        self.dropout = nn.Dropout(config.encoder.dropout)
        self.embeddings = nn.Embedding(
            config.dataset.vocab_size,
            config.encoder.embedding_size,
            padding_idx=config.dataset.pad_id,
        )
        self.lstm = nn.LSTM(
            input_size=config.encoder.embedding_size,
            hidden_size=config.encoder.hidden_size,
            bidirectional=config.encoder.bidirectional
        )
        self.fc = nn.Linear(config.encoder.hidden_size, config.num_classes)

    def forward(self, seq: torch.Tensor):
        out = self.embeddings(seq)
        out = self.dropout(out)
        lstm_out, (ht, ct) = self.lstm(out)
        return self.fc(ht[-1])
