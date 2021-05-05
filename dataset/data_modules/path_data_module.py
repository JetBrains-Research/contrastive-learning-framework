from os.path import join
from typing import Callable, Any, Optional, Tuple

import torch
from code2seq.dataset import PathContextBatch
from code2seq.utils.vocabulary import Vocabulary
from omegaconf import DictConfig

from dataset.base_data_module import BaseContrastiveDataModule
from dataset.classification_datasets import PathDataset


class PathDataModule(BaseContrastiveDataModule):
    def __init__(
            self,
            config: DictConfig,
            transform: Callable = None,
    ):
        BaseContrastiveDataModule.__init__(
            self,
            config=config,
            transform=transform,
        )
        self._vocabulary = None

        self._dataset_dir = join(config.data_folder, config.dataset.name, config.dataset.dir)
        self._train_data_file = join(self._dataset_dir, f"{config.dataset.name}.{config.train_holdout}.c2s")
        self._val_data_file = join(self._dataset_dir, f"{config.dataset.name}.{config.val_holdout}.c2s")
        self._test_data_file = join(self._dataset_dir, f"{config.dataset.name}.{config.test_holdout}.c2s")

        self.stage2path = {
            "train": self._train_data_file,
            "test": self._test_data_file,
            "val": self._val_data_file
        }

    def create_dataset(self, stage: str) -> Any:
        self._vocabulary = Vocabulary.load_vocabulary(
            join(
                self.config.data_folder,
                self.config.dataset.name,
                self.config.dataset.dir,
                self.config.vocabulary_name
            )
        )
        return PathDataset(self.stage2path[stage], self.config, self._vocabulary, False)

    def collate_fn(self, batch: Any) -> Any:
        a_pc = [sample["a_encoding"] for sample in batch]
        b_pc = [sample["b_encoding"] for sample in batch]
        labels = [sample["label"] for sample in batch]
        a_pc = PathContextBatch(a_pc)
        b_pc = PathContextBatch(b_pc)
        return (a_pc, b_pc), torch.LongTensor(labels)

    def transfer_batch_to_device(
            self,
            batch: Tuple[Tuple[PathContextBatch, PathContextBatch], torch.Tensor],
            device: Optional[torch.device] = None
    ) -> Tuple[Tuple[PathContextBatch, PathContextBatch], torch.Tensor]:
        (a, b), labels = batch
        a.move_to_device(device)
        b.move_to_device(device)
        labels.to(device)
        return (a, b), labels
