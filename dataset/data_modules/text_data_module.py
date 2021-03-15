from abc import abstractmethod
from typing import Any, Callable

import torch
from torch.nn.utils.rnn import pad_sequence

from dataset.base_data_module import BaseContrastiveDataModule
from dataset.classification_datasets.text_dataset import TextDataset


class TextDataModule(BaseContrastiveDataModule):
    def __init__(
            self,
            dataset_name: str,
            batch_size: int,
            num_classes: int,
            is_test: bool = False,
            transform: Callable = None
    ):
        super().__init__(
            dataset_name=dataset_name,
            batch_size=batch_size,
            is_test=is_test,
            transform=transform,
            num_classes=num_classes
        )

    @abstractmethod
    def create_dataset(self, dataset_path: str, stage: str) -> Any:
        return TextDataset(dataset_path=dataset_path, stage=stage, is_test=self.is_test)

    @abstractmethod
    def collate_fn(self, batch: Any) -> Any:
        # batch contains a list of tuples of structure (sequence, target)
        a = pad_sequence([item["a_encoding"].squeeze() for item in batch])
        b = pad_sequence([item["b_encoding"].squeeze() for item in batch])
        label = torch.LongTensor([item["label"] for item in batch])
        return (a, b), label

    @abstractmethod
    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        (a, b), label = batch
        a = a.to(device)
        b = b.to(device)
        if isinstance(label, torch.Tensor):
            label = label.to(device)
        return (a, b), label
