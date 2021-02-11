import tarfile
from os import listdir, rename
from os.path import join, exists

import wget
from tqdm import tqdm

poj_orig_link = "https://s3-eu-west-1.amazonaws.com/datasets.ml.labs.aws.intellij.net/poj-104/poj-104-original.tar.gz"
poj_orig_tar_path = join("data", "poj-104-original.tar.gz")
data_path = "data"
dataset_path = join(data_path, "poj_104")


def load_poj_104():
    if not exists(poj_orig_tar_path):
        print("Downloading dataset poj_104")
        wget.download(poj_orig_link, out=data_path)

    with tarfile.open(poj_orig_tar_path, "r:gz") as tar:
        print("Extracting files")
        tar.extractall(path=data_path)

    print("Renaming files")
    rename(join(data_path, "ProgramData"), dataset_path)
    for label_dir in tqdm(listdir(dataset_path)):
        label_dir_path = join(dataset_path, label_dir)
        for file_name in listdir(label_dir_path):
            name, _ = file_name.rsplit(".")
            new_file_name = f"{name}.cpp"
            rename(join(label_dir_path, file_name), join(label_dir_path, new_file_name))


if __name__ == "__main__":
    load_poj_104()
