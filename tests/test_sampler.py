from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10

from dataset.samplers import DiverseBatchSampler


class CustomDataset(Dataset):
    def __init__(self):
        self.dataset = CIFAR10(root="./data", train=False, download=True)

    def __getitem__(self, index):
        sample, label = self.dataset[index]
        return {"sample": sample, "label": label}

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    samples, labels = [], []
    for elem in batch:
        samples.append(elem["sample"])
        labels.append(elem["label"])
    return samples, labels


def test_contrastive_dataset():
    dataset = CustomDataset()
    batch_sampler = DiverseBatchSampler(dataset, batch_size=9, drop_last=True)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    for batch_ndx, (samples, labels) in enumerate(iter(dataloader)):
        if batch_ndx == 200:
            break
        assert len(labels) == len(set(labels))
