from itertools import combinations
from os import walk
from os.path import join, splitext
from typing import List, Dict

import torch
import youtokentome as yttm

from .base_dataset import BaseContrastiveDataset, FilesPair


class TextContrastiveDataset(BaseContrastiveDataset):
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)
        self.model_path = join(self.dataset_path, "model.yttm")
        self.tokenizer = yttm.BPE(model=self.model_path)

    @staticmethod
    def _check_if_cpp(file: str) -> bool:
        _, extension = splitext(file)
        return extension == ".cpp"

    def _get_pairs(self) -> Dict[str, List[FilesPair]]:
        _, base_dirs, _ = next(walk(self.dataset_path))
        base_dirs_paths = map(lambda file: join(self.dataset_path, file), base_dirs)
        pairs = {}
        for base_dir_path in base_dirs_paths:
            _, _, dir_files = next(walk(base_dir_path))
            dir_files_paths = map(lambda file: join(base_dir_path, file), dir_files)
            dir_files_paths = filter(self._check_if_cpp, dir_files_paths)
            pairs[base_dir_path] = list(combinations(dir_files_paths, 2))
        return pairs

    def _process_files(self, a_path: str, b_path: str) -> Dict:
        with open(a_path, "r") as a_file, open(b_path, "r") as b_file:
            a_text = a_file.read()
            b_text = b_file.read()
            return {
                "a_encoding": torch.LongTensor(self.tokenizer.encode([a_text])),
                "b_encoding": torch.LongTensor(self.tokenizer.encode([b_text])),
                "a_text": a_text,
                "b_text": b_text
            }
