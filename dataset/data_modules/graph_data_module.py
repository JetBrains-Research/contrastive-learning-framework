from collections.abc import Iterable
from typing import Any, Callable

import torch
from omegaconf import DictConfig
from torch_geometric.data import Batch

from dataset.base_data_module import BaseContrastiveDataModule
from dataset.classification_datasets.graph_dataset import GraphDataset


class GraphDataModule(BaseContrastiveDataModule):
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
        return GraphDataset(config=self.config, stage=stage)

    def collate_single_fn(self, batch: Any) -> Any:
        # batch contains a list of tuples of structure (sequence, target)
        inputs = Batch.from_data_list([item[0] for item in batch])
        labels = torch.LongTensor([item[1] for item in batch])
        return inputs, labels

    def collate_pair_fn(self, batch: Any) -> Any:
        # batch contains a list of tuples of structure (sequence, target)
        a = Batch.from_data_list([item["a_encoding"] for item in batch])
        b = Batch.from_data_list([item["b_encoding"] for item in batch])
        labels = torch.LongTensor([item["label"] for item in batch])
        return (a, b), labels

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        inputs, label = batch

        if isinstance(inputs, Batch):
            inputs = inputs.to(device)
        elif isinstance(inputs, Iterable):
            inputs = [
                input_.to(device) if input_ is not None else None for input_ in inputs
            ]
        else:
            raise ValueError(f"Unsupported type of inputs {type(inputs)}")

        if isinstance(label, torch.Tensor):
            label = label.to(device)
        return inputs, label
