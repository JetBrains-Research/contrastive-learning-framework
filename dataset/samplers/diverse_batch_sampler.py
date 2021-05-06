from collections import defaultdict
from typing import List

import numpy as np
from torch.utils.data.sampler import Sampler


class DiverseBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size, drop_last):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.label2idx = defaultdict(list)
        for idx in range(len(dataset)):
            self.label2idx[dataset[idx]["label"]].append(idx)

    def __iter__(self):
        n_samples_per_label = {
            label: len(ids) for label, ids in self.label2idx.items()
        }

        label_item_index = defaultdict(int)

        while len(n_samples_per_label.keys()):
            unused_labels = list(n_samples_per_label.keys())

            if len(unused_labels) >= self.batch_size:
                sampled_labels = np.random.choice(unused_labels, self.batch_size, replace=False)
            else:
                if self.drop_last:
                    break
                sampled_labels = np.random.choice(unused_labels, len(unused_labels), replace=False)

            batch = []
            for label in sampled_labels:
                cur_idx = label_item_index[label]
                batch.append(self.label2idx[label][cur_idx])
                label_item_index[label] += 1
                if label_item_index[label] == n_samples_per_label[label]:
                    label_item_index.pop(label)
                    n_samples_per_label.pop(label)
            if batch:
                np.random.shuffle(batch)
                yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
