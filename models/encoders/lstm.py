import torch
import torch.nn as nn

from configs import LSTMConfig


class LSTMModel(nn.Module):
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.num_classes = config.output_size
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.embedding_size,
            padding_idx=config.pad_id,
        )
        self.lstm = nn.LSTM(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            batch_first=config.batch_first,
            dropout=config.dropout,
            bias=config.bias,
            bidirectional=config.bidirectional
        )
        self.linear = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, seq: torch.Tensor):
        out = self.embeddings(seq)
        out = self.dropout(out)
        lstm_out, (ht, ct) = self.lstm(out)
        return self.linear(ht[-1])
