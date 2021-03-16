from os.path import join
from typing import Callable, Any, Optional, Tuple

import torch
from code2seq.dataset import PathContextBatch
from code2seq.utils.vocabulary import Vocabulary
from omegaconf import DictConfig, OmegaConf

from dataset.base_data_module import BaseContrastiveDataModule
from dataset.classification_datasets import PathDataset


def get_config() -> DictConfig:
    return OmegaConf.load("configs/code2class-poj104.yaml")


class PathDataModule(BaseContrastiveDataModule):
    def __init__(
            self,
            dataset_name: str,
            batch_size: int,
            num_classes: int,
            is_test: bool = False,
            transform: Callable = None,
    ):

        config = get_config()
        self._config = config
        self._vocabulary = Vocabulary.load_vocabulary(
            join(config.data_folder, config.dataset.name, config.vocabulary_name)
        )

        self._dataset_dir = join(config.data_folder, config.dataset.name)
        self._train_data_file = join(self._dataset_dir, f"{config.dataset.name}.{config.train_holdout}.c2s")
        self._val_data_file = join(self._dataset_dir, f"{config.dataset.name}.{config.val_holdout}.c2s")
        self._test_data_file = join(self._dataset_dir, f"{config.dataset.name}.{config.test_holdout}.c2s")

        self.stage2path = {
            "train": self._train_data_file,
            "test": self._test_data_file,
            "val": self._val_data_file
        }

        BaseContrastiveDataModule.__init__(
            self,
            dataset_name=dataset_name,
            batch_size=config.hyper_parameters.batch_size,
            is_test=is_test,
            transform=transform,
            num_classes=num_classes
        )

    def create_dataset(self, dataset_path: str, stage: str) -> Any:
        return PathDataset(self.stage2path[stage], self._config, self._vocabulary, False)

    def collate_fn(self, batch: Any) -> Any:
        a_pc = [sample["a_encoding"] for sample in batch]
        b_pc = [sample["b_encoding"] for sample in batch]
        labels = [sample["label"] for sample in batch]
        a_pc = PathContextBatch(a_pc)
        b_pc = PathContextBatch(b_pc)
        return (a_pc, b_pc), torch.LongTensor(labels)

    def transfer_batch_to_device(
            self, batch: Tuple[PathContextBatch, torch.Tensor], device: Optional[torch.device] = None
    ) -> Tuple[PathContextBatch, torch.Tensor]:
        pc, labels = batch
        if device is not None:
            pc.move_to_device(device)
            labels.to(device)
        return pc, labels
