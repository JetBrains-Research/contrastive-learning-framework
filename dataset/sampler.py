from collections import defaultdict
from copy import deepcopy
from random import shuffle
from typing import List

import numpy as np
from torch.utils.data.sampler import Sampler

from utils import get_task


class CodeforcesBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size, drop_last):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.task2idx = defaultdict(list)
        for idx in range(len(dataset)):
            _, label_idx = dataset[idx]
            label = dataset.encoding2label[label_idx]
            task = get_task(label)
            self.task2idx[task].append(idx)

    def __iter__(self):
        task2idx = deepcopy(self.task2idx)
        tasks = list(task2idx.keys())
        shuffle(tasks)

        while len(task2idx.keys()):
            task = np.random.choice(list(task2idx.keys()), replace=False)
            ids = task2idx[task]

            if len(ids) >= self.batch_size:
                sampled_ids = np.random.choice(ids, self.batch_size, replace=False)
            else:
                if self.drop_last:
                    break
                sampled_ids = deepcopy(ids)

            batch = []
            for sampled_id in sampled_ids:
                batch.append(sampled_id)
                ids.remove(sampled_id)

            np.random.shuffle(batch)
            yield batch

            if not ids:
                del task2idx[task]

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
