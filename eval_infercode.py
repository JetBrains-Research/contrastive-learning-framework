import pickle
from os import listdir
from os.path import join, dirname, basename

import torch

from models.self_supervised.utils import validation_metrics

data_dir = "data"


def compute_metrics(dataset):
    storage_path = join(data_dir, dataset, "infercode")
    labels, vectors = [], []
    for file in listdir(storage_path):
        file_path = join(storage_path, file)
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        labels += [basename(dirname(key)) for key in data]
        vectors += [torch.Tensor(data[key]).reshape(1, -1) for key in data]
    labels_set = set(labels)
    label2id = {label: i for i, label in enumerate(labels_set)}
    labels = torch.LongTensor([label2id[label] for label in labels])
    features = torch.cat(vectors, dim=0)
    log = validation_metrics([{"features": features, "labels": labels}], task=dataset)
    print(log)


if __name__ == "__main__":
    print("Codeforces:")
    compute_metrics("codeforces")
    print("POJ 104:")
    compute_metrics("poj_104")