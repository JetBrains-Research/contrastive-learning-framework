import tarfile
from argparse import ArgumentParser
from os import listdir, rename, mkdir
from os.path import join, exists
from shutil import rmtree

import splitfolders
import wget
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from preprocess import tokenize

poj_orig_link = "https://s3-eu-west-1.amazonaws.com/datasets.ml.labs.aws.intellij.net/poj-104/poj-104-original.tar.gz"
TRAIN_PART = 0.7
VAL_PART = 0.2
TEST_PART = 0.1


def load_poj_104(config: DictConfig):
    data_path = config.data_folder
    seed = config.seed
    orig_path = join(config.data_folder, "ProgramData")
    poj_orig_tar_path = join(config.data_folder, "poj-104-original.tar.gz")
    dataset_path = join(config.data_folder, config.dataset.name)
    if not exists(poj_orig_tar_path):
        print("Downloading dataset poj_104")
        wget.download(poj_orig_link, out=data_path)

    with tarfile.open(poj_orig_tar_path, "r:gz") as tar:
        print("Extracting files")
        tar.extractall(path=data_path)

    for label_dir in tqdm(listdir(orig_path)):
        label_dir_path = join(orig_path, label_dir)
        for file_name in listdir(label_dir_path):
            name, _ = file_name.rsplit(".")
            new_file_name = f"{name}.cpp"
            rename(join(label_dir_path, file_name), join(label_dir_path, new_file_name))

    mkdir(dataset_path)
    splitfolders.ratio(orig_path, output=dataset_path, seed=seed, ratio=(TRAIN_PART, VAL_PART, TEST_PART))
    rmtree(orig_path)

    tokenize(config)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config_path", type=str)
    args = arg_parser.parse_args()

    config_ = OmegaConf.load(args.config_path)
    load_poj_104(config_)
