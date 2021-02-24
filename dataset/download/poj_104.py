import tarfile
from os import listdir, rename, mkdir
from os.path import join, exists
from shutil import rmtree, move

import splitfolders
import wget
from tqdm import tqdm

from preprocess import tokenize

poj_orig_link = "https://s3-eu-west-1.amazonaws.com/datasets.ml.labs.aws.intellij.net/poj-104/poj-104-original.tar.gz"
poj_orig_tar_path = join("data", "poj-104-original.tar.gz")
data_path = "data"
orig_path = join(data_path, "ProgramData")
dataset_path = join(data_path, "poj_104")
TRAIN_PART = 0.7
VAL_PART = 0.2
TEST_PART = 0.1

SEED = 9


def load_poj_104():
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

    tokenize(orig_path)

    mkdir(dataset_path)
    move(join(orig_path, "model.yttm"), join(dataset_path, "model.yttm"))
    splitfolders.ratio(orig_path, output=dataset_path, seed=SEED, ratio=(TRAIN_PART, VAL_PART, TEST_PART))
    rmtree(orig_path)


if __name__ == "__main__":
    load_poj_104()
