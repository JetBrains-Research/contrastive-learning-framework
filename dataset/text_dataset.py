from os import walk
from os.path import join, splitext, basename, dirname
from typing import Tuple

import torch
import youtokentome as yttm
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, dataset_path: str):
        super().__init__()
        self.dataset_path = dataset_path
        self.model_path = join(self.dataset_path, "model.yttm")
        self.tokenizer = yttm.BPE(model=self.model_path)

        self.idx2file = dict()

        _, base_dirs, _ = next(walk(self.dataset_path))
        base_dirs_paths = map(lambda file: join(self.dataset_path, file), base_dirs)
        idx = 0
        for base_dir_path in base_dirs_paths:
            _, _, dir_files = next(walk(base_dir_path))
            dir_files_paths = map(lambda file: join(base_dir_path, file), dir_files)
            dir_files_paths = filter(self._in_c_family, dir_files_paths)
            for file in dir_files_paths:
                self.idx2file[idx] = file
                idx += 1

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
