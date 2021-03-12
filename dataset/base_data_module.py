from abc import abstractmethod
from os.path import exists
from os.path import join
from typing import Any, Callable

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .contrastive_dataset import ContrastiveDataset
from .download import load_dataset

SEED = 9


class BaseContrastiveDataModule(LightningDataModule):
    def __init__(
            self,
            dataset_name: str,
            num_classes: int,
            batch_size: int,
            is_test: bool = False,
            transform: Callable = None
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.dataset_path = join("data", dataset_name)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.transform = transform
        self.is_test = is_test

        self.clf_dataset = {}
        self.contrastive_dataset = {}

    @abstractmethod
    def create_dataset(self, dataset_path: str, stage: str) -> Any:
        pass

    def prepare_data(self):
        if not exists(self.dataset_path):
            load_dataset(self.dataset_name)

    def setup(self, stage: str = None):
        stages = []
        if stage == "fit" or stage is None:
            stages += ["train", "val"]
        if stage == "test" or stage is None:
            stages += ["test"]

        for stage in stages:
            self.clf_dataset[stage] = self.create_dataset(dataset_path=self.dataset_path, stage=stage)
            self.contrastive_dataset[stage] = ContrastiveDataset(clf_dataset=self.clf_dataset[stage])

    def train_dataloader(self):
        return DataLoader(
            self.contrastive_dataset["train"],
            batch_size=self.batch_size,
            collate_fn=self._collate,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.contrastive_dataset["val"],
            batch_size=self.batch_size,
            collate_fn=self._collate,
            shuffle=True,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.contrastive_dataset["test"],
            batch_size=self.batch_size,
            collate_fn=self._collate,
            shuffle=True,
            drop_last=True
        )

    @abstractmethod
    def collate_fn(self, batch: Any) -> Any:
        pass

    def _collate(self, batch: Any) -> Any:
        batch = self.collate_fn(batch)
        if self.transform is not None:
            return self.transform(batch)
        else:
            return batch
