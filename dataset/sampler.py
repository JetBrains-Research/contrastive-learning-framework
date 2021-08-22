from collections import defaultdict
from itertools import islice
from random import shuffle
from typing import List

from torch.utils.data.sampler import Sampler


class CodeforcesBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size, drop_last):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.task2idx = defaultdict(list)
        for idx in range(len(dataset)):
            label_idx = dataset[idx]["label"]
            label = dataset.clf_dataset.encoding2label[label_idx]
            task = self._get_task(label)
            self.task2idx[task].append(idx)

    def __iter__(self):
        batches = []

        for task in self.task2idx:
            shuffle(self.task2idx[task])
            batched_ids = list(self._chunk(self.task2idx[task], self.batch_size))

            if self.drop_last and (len(batched_ids[-1]) < self.batch_size):
                batched_ids = batched_ids[:-1]

            batches += batched_ids

        shuffle(batches)
        return iter(batches)

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    @staticmethod
    def _get_task(label: str):
        parts = label.split("_")
        return f"{parts[-6]}{parts[-5]}"

    @staticmethod
    def _chunk(it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())
