from pytorch_lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split, DataLoader

from .base_dataset import BaseDataset


class BaseDataModule(LightningDataModule):
    def __init__(self, dataset: str, batch_size: int):
        super().__init__()
        self.dataset = BaseDataset(dataset)
        self.batch_size = batch_size

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
        print([item["a_encoding"].shape for item in batch])
        a = pad_sequence([item["a_encoding"].squeeze() for item in batch], batch_first=True)
        b = pad_sequence([item["b_encoding"].squeeze() for item in batch], batch_first=True)
        return (a, b), None
