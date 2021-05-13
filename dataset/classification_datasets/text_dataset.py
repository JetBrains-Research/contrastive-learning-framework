from os.path import join, splitext, basename, dirname, exists
from random import shuffle
from typing import Tuple

import torch
import youtokentome as yttm
from omegaconf import DictConfig
from torch.utils.data import Dataset

from dataset.utils import traverse_clf_dataset
from preprocess import tokenize


class TextDataset(Dataset):
    def __init__(self, config: DictConfig, stage: str):
        super().__init__()

        self.dataset_path = join(config.data_folder, config.dataset.name, config.dataset.dir)
        self.tokenizer_name = config.dataset.tokenizer_name
        self.data_path = join(self.dataset_path, stage)
        self.max_seq_len = config.encoder.max_seq_len

        self.tokenizer = self._get_tokenizer(config=config)

        self.files = traverse_clf_dataset(self.data_path)
        shuffle(self.files)

    def _get_tokenizer(self, config: DictConfig) -> yttm.BPE:
        model_path = join(self.dataset_path, self.tokenizer_name)
        if not exists(model_path):
            tokenize(config=config)
        return yttm.BPE(model=model_path)

    @staticmethod
    def _in_c_family(file: str) -> bool:
        _, extension = splitext(file)
        return (extension == ".cpp") or (extension == ".c")

    def _process_file(self, path: str) -> Tuple:
        with open(path, "r", encoding="utf8", errors='ignore') as file:
            text = file.read()
            encoding = self.tokenizer.encode([text], bos=True)
            encoding = [encoding[0][:self.max_seq_len]]
            return torch.LongTensor(encoding), basename(dirname(path))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        return self._process_file(self.files[idx])
