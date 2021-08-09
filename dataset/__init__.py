from dataset.classification_datasets import TextDataset, PathDataset, GraphDataset
from dataset.data_modules import TextDataModule, PathDataModule, GraphDataModule, CodeTransformerModule
from .base_data_module import BaseContrastiveDataModule
from .contrastive_dataset import ContrastiveDataset

__all__ = [
    "TextDataset",
    "TextDataModule",
    "PathDataset",
    "PathDataModule",
    "GraphDataset",
    "GraphDataModule",
    "CodeTransformerModule",
    "ContrastiveDataset",
    "BaseContrastiveDataModule",
    "data_modules"
]

data_modules = {
    "transformer": TextDataModule,
    "code2class": PathDataModule,
    "gnn": GraphDataModule,
    "code-transformer": CodeTransformerModule
}
