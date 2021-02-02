from abc import abstractmethod
from typing import List, Tuple, Dict

from torch.utils.data import Dataset

FilesPair = Tuple[str, str]


class BaseContrastiveDataset(Dataset):
    def __init__(self, dataset_path: str):
        super().__init__()
        self.dataset_path = dataset_path
        self.pairs = self._get_pairs()

    @abstractmethod
    def _get_pairs(self) -> Dict[str, List[FilesPair]]:
        pass

    @abstractmethod
    def _process_files(self, a_path: str, b_path: str) -> Dict:
        pass

    def __len__(self) -> int:
        return sum([len(v) for k, v in self.pairs.items()])

    def __getitem__(self, idx: int) -> Dict:
        pair_idx = 0
        for file, pairs in self.pairs.items():
            if pair_idx + len(pairs) > idx:
                a_path, b_path = pairs[idx - pair_idx]
                return self._process_files(a_path, b_path)
            pair_idx += len(pairs)
        return {}
