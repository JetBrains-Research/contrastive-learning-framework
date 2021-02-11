from torchvision.datasets import MNIST

from dataset.contrastive_dataset import ContrastiveDataset


def test_contrastive_dataset():
    clf_dataset = MNIST(root="./data", train=False, download=True)
    contr_dataset = ContrastiveDataset(clf_dataset=clf_dataset)
    for _, (a_idx, b_idx) in contr_dataset.idx2pair.items():
        _, label_a = clf_dataset[a_idx]
        _, label_b = clf_dataset[b_idx]
        assert label_a == label_b
