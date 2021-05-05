from typing import Any, Callable

import torch
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence

from dataset.base_data_module import BaseContrastiveDataModule
from dataset.classification_datasets.text_dataset import TextDataset


class TextDataModule(BaseContrastiveDataModule):
    def __init__(
            self,
            config: DictConfig,
            transform: Callable = None
    ):
        super().__init__(
            config=config,
            transform=transform,
        )

    def create_dataset(self, stage: str) -> Any:
        return TextDataset(config=self.config, stage=stage)

    def collate_fn(self, batch: Any) -> Any:
        # batch contains a list of tuples of structure (sequence, target)
        a = pad_sequence([item["a_encoding"].squeeze() for item in batch])
        b = pad_sequence([item["b_encoding"].squeeze() for item in batch])
        label = torch.LongTensor([item["label"] for item in batch])
        return (a, b), label

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        inputs, label = batch
        inputs = [
            input_.to(device) if input_ is not None else None for input_ in inputs
        ]
        if isinstance(label, torch.Tensor):
            label = label.to(device)
        return inputs, label
