from os.path import join, splitext, basename, dirname, exists
from typing import Tuple

import torch
import youtokentome as yttm
from torch.utils.data import Dataset

from dataset.utils import traverse_clf_dataset
from preprocess import tokenize


class TextDataset(Dataset):
    def __init__(self, dataset_path: str, stage: str, is_test: bool = False):
        super().__init__()

        self.dataset_path = dataset_path
        self.data_path = join(dataset_path, stage)
        self.is_test = is_test

        self.tokenizer = self._get_tokenizer()

        self.idx2file = traverse_clf_dataset(self.data_path)

    def _get_tokenizer(self) -> yttm.BPE:
        model_path = join(self.dataset_path, "model.yttm")
        if not exists(model_path):
            tokenize(dataset_path=self.dataset_path)
        return yttm.BPE(model=model_path)

    @staticmethod
    def _in_c_family(file: str) -> bool:
        _, extension = splitext(file)
        return (extension == ".cpp") or (extension == ".c")

    def _process_file(self, path: str) -> Tuple:
        with open(path, "r", encoding="utf8", errors='ignore') as file:
            text = file.read()
            encoding = self.tokenizer.encode([text])
            return torch.LongTensor(encoding), basename(dirname(path))

    def __len__(self):
        return len(self.idx2file)

    def __getitem__(self, idx: int):
        return self._process_file(self.idx2file[idx])
