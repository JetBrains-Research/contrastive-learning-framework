from itertools import combinations
from os import listdir
from os.path import join, isdir, splitext

import torch
import youtokentome as yttm
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, dataset: str):
        super().__init__()
        dataset_path = join("data", dataset)
        model_path = join(dataset_path, "model.yttm")
        self.tokenizer = yttm.BPE(model=model_path)
        self.files = [
            join(dataset_path, elem) for elem in listdir(dataset_path) if isdir(join(dataset_path, elem))
        ]
        self.pairs = {}
        for file in self.files:
            transformed_files = [
                join(file, elem) for elem in listdir(file) if self._check_if_cpp(join(file, elem))
            ]
            self.pairs[file] = list(combinations(transformed_files, 2))

    @staticmethod
    def _check_if_cpp(file: str):
        _, extension = splitext(file)
        return extension == ".cpp"

    def __len__(self):
        return sum([len(v) for k, v in self.pairs.items()])

    def __getitem__(self, idx):
        pair_idx = 0
        for file, pairs in self.pairs.items():
            if pair_idx + len(pairs) > idx:
                a_path, b_path = pairs[idx - pair_idx]
                with open(a_path, "r") as a_file, open(b_path, "r") as b_file:
                    a_text = a_file.read()
                    b_text = b_file.read()
                    return {
                        "file": file,
                        "a_encoding": torch.LongTensor(self.tokenizer.encode([a_text])),
                        "b_encoding": torch.LongTensor(self.tokenizer.encode([b_text])),
                        "a_text": a_text,
                        "b_text": b_text
                    }
            pair_idx += len(pairs)
