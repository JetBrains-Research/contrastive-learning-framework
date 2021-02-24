from os import mkdir
from os.path import exists

from .poj_104 import load_poj_104

dataset2script = {
    "poj_104": load_poj_104
}

data_dir = "data"


def load_dataset(dataset_name: str):
    if not exists(data_dir):
        mkdir(data_dir)
    dataset2script[dataset_name]()
