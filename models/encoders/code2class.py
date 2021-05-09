from code2seq.dataset import PathContextBatch
from code2seq.model.modules import PathEncoder, PathClassifier
from code2seq.utils.vocabulary import Vocabulary, PAD
from omegaconf import DictConfig
from torch import nn


class Code2ClassModel(nn.Module):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary):
        super().__init__()
        self.num_classes = config.num_classes
        self.encoder = PathEncoder(
            config.encoder,
            config.classifier.classifier_input_size,
            len(vocabulary.token_to_id),
            vocabulary.token_to_id[PAD],
            len(vocabulary.node_to_id),
            vocabulary.node_to_id[PAD],
        )
        self.classifier = PathClassifier(config.classifier, config.num_classes)

    def forward(self, batch: PathContextBatch):
        return self.classifier(self.encoder(batch.contexts), batch.contexts_per_label)
