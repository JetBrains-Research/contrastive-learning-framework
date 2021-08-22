from collections.abc import Iterable
from os.path import join
from typing import Callable, Any

import torch
from code2seq.data.path_context import BatchedLabeledPathContext
from code2seq.data.vocabulary import Vocabulary
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
        self._vocabulary = Vocabulary(
            join(
                self.config.data_folder,
                self.config.dataset.name,
                self.config.dataset.dir,
                self.config.vocabulary_name
            ),
            self.config.dataset.max_labels,
            self.config.dataset.max_tokens
        )
        return PathDataset(self.stage2path[stage], self.config.dataset, self._vocabulary, False)

    def collate_single_fn(self, batch: Any) -> Any:
        pc = BatchedLabeledPathContext([sample[0] for sample in batch])
        labels = torch.LongTensor([sample[1] for sample in batch])
        return pc, labels

    def collate_pair_fn(self, batch: Any) -> Any:
        a_pc = [sample["a_encoding"] for sample in batch]
        b_pc = [sample["b_encoding"] for sample in batch]
        a_pc = BatchedLabeledPathContext(a_pc)
        b_pc = BatchedLabeledPathContext(b_pc)
        labels = torch.LongTensor([sample["label"] for sample in batch])
        return (a_pc, b_pc), labels

    def transfer_batch_to_device(self, batch, device: torch.device, **kwargs):
        inputs, labels = batch

        if isinstance(inputs, BatchedLabeledPathContext):
            inputs.move_to_device(device)
        elif isinstance(inputs, Iterable):
            for input_ in inputs:
                if input_ is not None:
                    input_.move_to_device(device)
        else:
            raise ValueError(f"Unsupported type of inputs {type(inputs)}")

        labels = labels.to(device)
        return inputs, labels
