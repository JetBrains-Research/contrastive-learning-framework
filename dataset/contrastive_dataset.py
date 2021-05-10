from collections import defaultdict
from itertools import combinations
from typing import Tuple, Dict

from torch.utils.data import Dataset
from tqdm import tqdm

Pair = Tuple[str, str]


class ContrastiveDataset(Dataset):
    def __init__(self, clf_dataset: Dataset):
        super().__init__()
        self.clf_dataset = clf_dataset

        # Mapping from label to idx in clf_dataset
        self.label2idx = defaultdict(list)
        for clf_idx in tqdm(range(len(self.clf_dataset))):
            _, label = self.clf_dataset[clf_idx]
            self.label2idx[label].append(clf_idx)
        # Pairs of indexes related to elements of clf_dataset having the same class
        label2pairs = {k: list(combinations(v, 2)) for k, v in self.label2idx.items() if len(v) > 1}
        self.idx2label = dict()
        for label, idxs in self.label2idx.items():
            for idx in idxs:
                self.idx2label[idx] = label

        # Mapping from index in contrastive to pair of indexes in clf_dataset related to the same class
        self.idx2pair = dict()
        contrastive_idx = 0
        for label, pairs in label2pairs.items():
            for pair_idx, pair in enumerate(pairs):
                self.idx2pair[contrastive_idx + pair_idx] = pair
            contrastive_idx += len(pairs)

        # Encoding labels
        self.label2encoding = {
            label: label_idx for label_idx, label in enumerate(label2pairs.keys())
        }

    def __len__(self) -> int:
        return len(self.idx2pair)

    def __getitem__(self, idx: int) -> Dict:
        a_idx, b_idx = self.idx2pair[idx]
        a_encoding, _ = self.clf_dataset[a_idx]
        b_encoding, _ = self.clf_dataset[b_idx]
        return {
            "a_encoding": a_encoding,
            "b_encoding": b_encoding,
            "label": self.label2encoding[self.idx2label[a_idx]]
        }
