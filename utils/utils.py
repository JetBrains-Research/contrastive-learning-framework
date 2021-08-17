from itertools import islice
from os.path import isfile

from omegaconf import DictConfig


def is_json_file(path: str):
    ext = path.rsplit(".", 1)[-1]
    return isfile(path) and (ext == "json")


def get_task(label: str):
    parts = label.split("_")
    return f"{parts[-6]}{parts[-5]}"


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def replace_str_none(cfg):
    for k, v in cfg.items():
        if isinstance(v, DictConfig):
            replace_str_none(v)
        else:
            if v == "None":
                cfg[k] = None
    return cfg
