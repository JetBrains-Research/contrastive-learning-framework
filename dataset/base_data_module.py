from abc import abstractmethod
from os.path import join
from typing import Any, Callable

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, BatchSampler, RandomSampler

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
        self.train_batch_size = config.hyper_parameters.batch_size
        self.test_batch_size = config.hyper_parameters.test_batch_size
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
        load_dataset(self.config)

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
            batch_size=self.train_batch_size,
            drop_last=True,
            shuffle=True,
            collate_fn=self._collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.contrastive_dataset[self.val_holdout],
            batch_size=self.test_batch_size,
            drop_last=True,
            shuffle=True,
            collate_fn=self._collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.contrastive_dataset[self.test_holdout],
            batch_size=self.test_batch_size,
            drop_last=True,
            shuffle=True,
            collate_fn=self._collate,
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
