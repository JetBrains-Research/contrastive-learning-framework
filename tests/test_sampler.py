from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10

from dataset import ContrastiveDataset
from dataset.sampler import CodeforcesBatchSampler


class WrappedDataset(Dataset):
    def __init__(self):
        self.dataset = CIFAR10(root="./data", train=False, download=True)
        self.encoding2label = dict()
        for label in range(10):
            self.encoding2label[label] = f"test_CF_user_{label}_F_12345678_tests_OK_1234"

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return 1000


def collate_fn(batch):
    labels = []
    for data in batch:
        labels.append(data["label"])
    return None, labels


def test_codeforces_batch_sampler():
    dataset = WrappedDataset()
    contr_dataset = ContrastiveDataset(clf_dataset=dataset)
    batch_sampler = CodeforcesBatchSampler(contr_dataset, batch_size=8, drop_last=True)
    dataloader = DataLoader(contr_dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    for batch_idx, (_, labels) in enumerate(iter(dataloader)):
        if batch_idx == 200:
            break
        assert len(set(labels)) == 1
