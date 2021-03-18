from abc import abstractmethod
from os.path import exists
from os.path import join
from typing import Any, Callable

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .contrastive_dataset import ContrastiveDataset
from .download import load_dataset

SEED = 9


class BaseContrastiveDataModule(LightningDataModule):
    def __init__(
            self,
            config: DictConfig,
            transform: Callable = None
    ):
        super().__init__()

        self.config = config
        self.dataset_name = config.dataset.name
        self.dataset_path = join(config.data_folder, self.dataset_name)
        self.batch_size = config.hyper_parameters.batch_size
        self.num_classes = config.num_classes
        self.transform = transform

        self.train_holdout = config.train_holdout
        self.test_holdout = config.test_holdout
        self.val_holdout = config.val_holdout

        self.clf_dataset = {}
        self.contrastive_dataset = {}

    @abstractmethod
    def create_dataset(self, stage: str) -> Any:
        pass

    def prepare_data(self):
        if not exists(self.dataset_path):
            load_dataset(self.dataset_name)

    def setup(self, stage: str = None):
        stages = []
        if stage == "fit" or stage is None:
            stages += [self.train_holdout, self.val_holdout]
        if stage == "test" or stage is None:
            stages += [self.test_holdout]

        for stage in stages:
            self.clf_dataset[stage] = self.create_dataset(stage=stage)
            self.contrastive_dataset[stage] = ContrastiveDataset(clf_dataset=self.clf_dataset[stage])

    def train_dataloader(self):
        return DataLoader(
            self.contrastive_dataset[self.train_holdout],
            batch_size=self.batch_size,
            collate_fn=self._collate,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.contrastive_dataset[self.val_holdout],
            batch_size=self.batch_size,
            collate_fn=self._collate,
            shuffle=True,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.contrastive_dataset[self.test_holdout],
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
