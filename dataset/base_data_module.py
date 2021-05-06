from abc import abstractmethod
from os.path import join
from typing import Any, Callable

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, BatchSampler, RandomSampler

from .contrastive_dataset import ContrastiveDataset
from .download import load_dataset
from .samplers import DiverseBatchSampler

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

    def get_batch_sampler(self, dataset, batch_size, drop_last):
        if self.config.ssl.name in ["MocoV2", "BYOL"]:
            return BatchSampler(sampler=RandomSampler(dataset), batch_size=batch_size, drop_last=drop_last)
        elif self.config.ssl.name == "SimCLR":
            return DiverseBatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                drop_last=drop_last
            )
        else:
            raise ValueError(f"Unknown ssl method {self.config.ssl.name}")

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
        dataset = self.contrastive_dataset[self.train_holdout]
        return DataLoader(
            dataset,
            batch_sampler=self.get_batch_sampler(dataset, batch_size=self.batch_size, drop_last=True),
            collate_fn=self._collate,
        )

    def val_dataloader(self):
        dataset = self.contrastive_dataset[self.val_holdout]
        return DataLoader(
            dataset,
            batch_sampler=self.get_batch_sampler(dataset, batch_size=self.batch_size, drop_last=True),
            collate_fn=self._collate,
        )

    def test_dataloader(self):
        dataset = self.contrastive_dataset[self.test_holdout]
        return DataLoader(
            dataset,
            batch_sampler=self.get_batch_sampler(dataset, batch_size=self.batch_size, drop_last=True),
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
