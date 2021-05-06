from code2seq.dataset import PathContextBatch
from code2seq.model import Code2Class
from code2seq.utils.vocabulary import Vocabulary
from omegaconf import DictConfig
from torch import nn


class Code2ClassModel(nn.Module):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary):
        super().__init__()
        self.num_classes = config.num_classes
        self.code2class = Code2Class(config, vocabulary)

    def forward(self, batch: PathContextBatch):
        out = self.code2class(batch.contexts, batch.contexts_per_label)
        return out
