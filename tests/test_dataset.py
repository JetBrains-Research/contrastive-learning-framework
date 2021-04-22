import subprocess
import sys
from collections import defaultdict
from os import listdir
from os.path import join, exists

from torchvision.datasets import CIFAR10

from dataset.contrastive_dataset import ContrastiveDataset


def test_contrastive_dataset():
    clf_dataset = CIFAR10(root="./data", train=False, download=True)
    contr_dataset = ContrastiveDataset(clf_dataset=clf_dataset)
    for i, item in enumerate(contr_dataset.idx2pair.items()):
        if i == 200:
            break
        _, (a_idx, b_idx) = item
        _, label_a = clf_dataset[a_idx]
        _, label_b = clf_dataset[b_idx]
        assert label_a == label_b


def test_split_poj_104():
    for dataset in ["poj_104", "codeforces"]:
        subprocess.run(
            args=[
                "sh",
                join("scripts", "download", "download_data.sh"),
                "--dataset", dataset,
                "--dev",
            ],
            stderr=sys.stderr,
            stdout=sys.stdout
        )

        dataset_path = join("data", dataset, "raw")
        assert exists(dataset_path)

        holdout2class = defaultdict(set)
        for holdout in ["train", "test", "val"]:
            holdout_path = join(dataset_path, holdout)
            for dir_ in listdir(holdout_path):
                if dataset == "codeforces":
                    parts = dir_.split("_")
                    task = f"{parts[-6]}{parts[-5]}"
                    holdout2class[holdout].add(task)
                else:
                    holdout2class[holdout].add(dir_)

        assert not (holdout2class["train"] & holdout2class["test"]), f"train and test intersect in {dataset}"
        assert not (holdout2class["val"] & holdout2class["test"]), f"val and test intersect in {dataset}"
        assert not (holdout2class["train"] & holdout2class["val"]), f"train and val intersect in {dataset}"
