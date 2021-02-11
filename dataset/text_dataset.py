from os import walk
from os.path import join, splitext, basename, dirname, exists
from typing import Tuple

import torch
import youtokentome as yttm
from torch.utils.data import Dataset
from preprocess import tokenize


class TextDataset(Dataset):
    def __init__(self, dataset_path: str, is_test: bool = False):
        super().__init__()

        self.dataset_path = dataset_path
        model_path = join(self.dataset_path, "model.yttm")
        self.is_test = is_test

        self.tokenizer = self._get_tokenizer(model_path)

        self.idx2file = dict()

        _, base_dirs, _ = next(walk(self.dataset_path))
        base_dirs_paths = map(lambda file_: join(self.dataset_path, file_), base_dirs)
        idx = 0
        for base_dir_path in base_dirs_paths:
            _, _, dir_files = next(walk(base_dir_path))
            dir_files_paths = map(lambda file_: join(base_dir_path, file_), dir_files)
            dir_files_paths = filter(self._in_c_family, dir_files_paths)
            for file in dir_files_paths:
                self.idx2file[idx] = file
                idx += 1

    def _get_tokenizer(self, model_path: str) -> yttm.BPE:
        if exists(model_path):
            return yttm.BPE(model=model_path)
        else:
            return tokenize(
                dataset_path=self.dataset_path,
                model_path=model_path,
                is_test=self.is_test
            )

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
