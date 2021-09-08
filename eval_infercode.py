import pickle
from itertools import combinations
from os import listdir
from os.path import join, dirname, basename

import numpy as np
import torch
import yaml

from models.self_supervised.utils import compute_map_at_k

data_dir = "data"


def compute_metrics(dataset, ks):
    storage_path = join(data_dir, dataset, "infercode")
    vectors = []
    for i, file in listdir(storage_path):
        file_path = join(storage_path, file)
        with open(file_path, "rb") as f:
            vectors[i] = pickle.load(f)
    print(vectors[0])


if __name__ == "__main__":
    print("Codeforces:")
    compute_metrics("codeforces", ks=(5, 10, 15))
    print("POJ 104:")
    compute_metrics("poj_104", ks=(100, 200, 500))
