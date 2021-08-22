from code2seq.data.path_context import BatchedLabeledPathContext
from code2seq.model.modules import PathEncoder
from commode_utils.modules import Classifier
from code2seq.data.vocabulary import Vocabulary
from omegaconf import DictConfig
from torch import nn


class Code2ClassModel(nn.Module):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary):
        super().__init__()
        self.num_classes = config.num_classes
        self.encoder = PathEncoder(
            config=config.encoder,
            n_tokens=len(vocabulary.token_to_id),
            token_pad_id=vocabulary.token_to_id[vocabulary.PAD],
            n_nodes=len(vocabulary.node_to_id),
            node_pad_id=vocabulary.node_to_id[vocabulary.PAD],
        )
        self.classifier = Classifier(config.classifier, config.num_classes)

    def forward(self, batch: BatchedLabeledPathContext):
        return self.classifier(
            self.encoder(
                batch.from_token,
                batch.path_nodes,
                batch.to_token
            ),
            batch.contexts_per_label
        )
