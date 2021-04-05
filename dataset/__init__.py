from dataset.classification_datasets.text_dataset import TextDataset
from dataset.data_modules import TextDataModule, PathDataModule, GraphDataModule
from .base_data_module import BaseContrastiveDataModule
from .contrastive_dataset import ContrastiveDataset

__all__ = [
    "TextDataset",
    "TextDataModule",
    "PathDataModule",
    "GraphDataModule",
    "ContrastiveDataset",
    "BaseContrastiveDataModule",
    "data_modules"
]

data_modules = {
    "lstm": TextDataModule,
    "code2class": PathDataModule,
    "gnn": GraphDataModule
}
