from pytorch_lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split, DataLoader
import torch
from typing import Dict
from os.path import exists
from .text_dataset import TextContrastiveDataset


class BaseDataModule(LightningDataModule):
    def __init__(self, dataset_path: str, batch_size: int):
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset = TextContrastiveDataset(dataset_path)
        self.batch_size = batch_size

    def prepare_data(self):
        if not exists(self.dataset_path):
            raise ValueError(f"There is no file in passed path ({self.dataset_path})")

    def setup(self, stage: str = None):
        len_train = int(0.6 * len(self.dataset))
        len_test = int(0.2 * len(self.dataset))
        len_val = len(self.dataset) - len_test - len_train
        self.train, self.val, self.test = random_split(self.dataset, [len_train, len_val, len_test])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=self._collate)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=self._collate)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, collate_fn=self._collate)

    @staticmethod
    def _collate(batch):
        # batch contains a list of tuples of structure (sequence, target)
        a = pad_sequence([item["a_encoding"].squeeze() for item in batch], batch_first=True)
        b = pad_sequence([item["b_encoding"].squeeze() for item in batch], batch_first=True)
        return (a, b), None

    def transfer_batch_to_device(self, batch: Dict, device: torch.device) -> Dict:
        for key in ["a_encoding", "b_encoding"]:
            batch[key] = batch[key].to(device)
        return batch
