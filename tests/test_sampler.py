from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10

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
        return len(self.dataset)


def collate_fn(batch):
    samples, labels = [], []
    for sample, label in batch:
        samples.append(sample)
        labels.append(label)
    return samples, labels


def test_contrastive_dataset():
    dataset = WrappedDataset()
    batch_sampler = CodeforcesBatchSampler(dataset, batch_size=80, drop_last=True)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    for batch_idx, (samples, labels) in enumerate(iter(dataloader)):
        if batch_idx == 200:
            break
        assert len(set(labels)) == 1
