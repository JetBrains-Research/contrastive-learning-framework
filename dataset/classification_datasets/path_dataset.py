from typing import Optional, Tuple, Any

from code2seq.dataset import PathContextDataset, PathContextSample
from code2seq.utils.vocabulary import Vocabulary
from omegaconf import DictConfig


class PathDataset(PathContextDataset):
    def __init__(self, data_file_path: str, config: DictConfig, vocabulary: Vocabulary, random_context: bool):
        super().__init__(data_file_path, config, vocabulary, random_context)
        self.encoding2label = {
            label_id: label for label, label_id in vocabulary.label_to_id.items()
        }

    def __getitem__(self, index) -> Optional[Tuple[PathContextSample, Any]]:
        pcs = super().__getitem__(index)
        return pcs, pcs.label[0][0]
