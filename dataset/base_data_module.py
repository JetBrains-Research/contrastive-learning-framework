from abc import abstractmethod
from os.path import join
from typing import Any, Callable

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, BatchSampler, RandomSampler

from .contrastive_dataset import ContrastiveDataset
from .download import load_dataset
from .sampler import CodeforcesBatchSampler

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

    def get_batch_sampler(self, dataset, batch_size, drop_last):
        if self.config.dataset.name == "poj_104":
            return BatchSampler(sampler=RandomSampler(dataset), batch_size=batch_size, drop_last=drop_last)
        elif self.config.dataset.name == "codeforces":
            return CodeforcesBatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                drop_last=drop_last
            )
        else:
            raise ValueError(f"Unknown ssl method {self.config.ssl.name}")

    def setup(self, stage: str = None):
        assert stage is not None

        stages = []
        if stage == "fit":
            stages += [self.train_holdout, self.val_holdout]
        if stage == "test":
            stages += [self.test_holdout]

        for stage_ in stages:
            self.clf_dataset[stage_] = self.create_dataset(stage=stage_)

        if stage == "fit":
            self.contrastive_dataset[self.train_holdout] = ContrastiveDataset(
                clf_dataset=self.clf_dataset[self.train_holdout]
            )

    def train_dataloader(self):
        dataset = self.contrastive_dataset[self.train_holdout]
        return DataLoader(
            dataset=dataset,
            batch_sampler=self.get_batch_sampler(
                dataset=dataset,
                batch_size=self.train_batch_size,
                drop_last=True
            ),
            collate_fn=self._collate_contrastive,
        )

    def val_dataloader(self):
        return DataLoader(
            self.clf_dataset[self.val_holdout],
            batch_size=self.test_batch_size,
            collate_fn=self._collate_single,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.clf_dataset[self.test_holdout],
            batch_size=self.test_batch_size,
            collate_fn=self._collate_single,
            shuffle=True,
        )

    @abstractmethod
    def collate_single_fn(self, batch: Any) -> Any:
        pass

    def _collate_single(self, batch: Any) -> Any:
        return self.collate_single_fn(batch)

    @abstractmethod
    def collate_pair_fn(self, batch: Any) -> Any:
        pass

    def _collate_contrastive(self, batch: Any) -> Any:
        batch = self.collate_pair_fn(batch)
        if self.transform is not None:
            return self.transform(batch)
        else:
            return batch
