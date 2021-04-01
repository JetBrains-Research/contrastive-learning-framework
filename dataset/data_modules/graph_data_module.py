from typing import Any, Callable

import torch
from omegaconf import DictConfig
from torch_geometric.data import Batch
from torch_geometric.data.dataloader import Collater

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
        self.collater = Collater(follow_batch=[])

    def create_dataset(self, stage: str) -> Any:
        return GraphDataset(config=self.config, stage=stage)

    def collate_fn(self, batch: Any) -> Any:
        # batch contains a list of tuples of structure (sequence, target)
        a = Batch.from_data_list([item["a_encoding"]for item in batch])
        b = Batch.from_data_list([item["b_encoding"] for item in batch])
        label = torch.LongTensor([item["label"] for item in batch])
        return (a, b), label

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        (a, b), label = batch
        a = a.to(device)
        b = b.to(device)
        if isinstance(label, torch.Tensor):
            label = label.to(device)
        return (a, b), label
