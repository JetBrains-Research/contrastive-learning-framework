from collections import defaultdict
from random import shuffle
from typing import List

from torch.utils.data.sampler import Sampler

from utils import get_task, chunk


class CodeforcesBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size, drop_last):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.task2idx = defaultdict(list)
        for idx in range(len(dataset)):
            label_idx = dataset[idx]["label"]
            label = dataset.clf_dataset.encoding2label[label_idx]
            task = get_task(label)
            self.task2idx[task].append(idx)

    def __iter__(self):
        batches = []

        for task in self.task2idx:
            shuffle(self.task2idx[task])
            batched_ids = list(chunk(self.task2idx[task], self.batch_size))

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
