from os import mkdir
from os.path import exists

from omegaconf import DictConfig

from .poj_104 import load_poj_104

dataset2script = {
    "poj_104": load_poj_104
}


def load_dataset(config: DictConfig):
    if not exists(config.data_folder):
        mkdir(config.data_folder)
    dataset2script[config.dataset.name](config)
