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
