from os import walk
from os.path import exists
from os.path import join
from typing import Any, Callable

import torch
from pytorch_lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .contrastive_dataset import ContrastiveDataset
from .download import load_dataset
from .text_dataset import TextDataset

encoder2datasets = {
    "LSTM": TextDataset,
}

SEED = 9


class BaseDataModule(LightningDataModule):
    def __init__(self, encoder_name: str, dataset_name: str, batch_size: int, is_test: bool = False, transform: Callable = None):
        super().__init__()
        if encoder_name in encoder2datasets:
            self.dataset = encoder2datasets[encoder_name]
        else:
            raise NotImplementedError(f"Dataset for {encoder_name} is currently not available")

        self.dataset_name = dataset_name
        self.dataset_path = join("data", dataset_name)
        self.batch_size = batch_size
        _, base_dirs, _ = next(walk(join(self.dataset_path, "train")))
        self.num_classes = len(base_dirs)
        self.transform = transform
        self.is_test = is_test

        self.clf_dataset = {}
        self.contrastive_dataset = {}

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
            self.clf_dataset[stage] = self.dataset(dataset_path=self.dataset_path, stage=stage, is_test=self.is_test)
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

    def _collate(self, batch):
        # batch contains a list of tuples of structure (sequence, target)
        a = pad_sequence([item["a_encoding"].squeeze() for item in batch], batch_first=True)
        b = pad_sequence([item["b_encoding"].squeeze() for item in batch], batch_first=True)
        label = torch.LongTensor([item["label"] for item in batch])
        data = (a, b), label

        if self.transform is not None:
            return self.transform(data)
        else:
            return data

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        (a, b), label = batch
        a = a.to(device)
        b = b.to(device)
        if isinstance(label, torch.Tensor):
            label = label.to(device)
        return (a, b), label
