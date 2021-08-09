from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm


class CodeTransformerDataset(Dataset):
    def __init__(self, iter_dataset: IterableDataset):
        super().__init__()

        self.samples = []
        list_dataset = [next(iter_dataset)]
        num_samples = len(list_dataset)
        print(num_samples)

        for sample in tqdm(list_dataset, total=num_samples):
            self.samples.append(sample)

        # Encoding labels
        self.labels = set([sample.func_name for sample in self.samples])
        self.label2encoding = {
            label: label_idx for label_idx, label in enumerate(self.labels)
        }
        self.encoding2label = {
            label_idx: label for label_idx, label in enumerate(self.labels)
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        return sample, self.label2encoding[sample.func_name]
