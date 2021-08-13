from collections.abc import Iterable
from typing import Any, Callable

import torch
from code_transformer.experiments.mixins.code_summarization import CTCodeSummarizationMixin
from code_transformer.preprocessing.datamanager.base import CTBatch
from omegaconf import DictConfig, OmegaConf
from dataset.classification_datasets import CodeTransformerDataset

from dataset.contrastive_dataset import ContrastiveDataset
from dataset.base_data_module import BaseContrastiveDataModule
from code_transformer.experiments.experiment import ExperimentSetup
from code_transformer.preprocessing.datamanager.base import batch_to_device

from dataset.download import load_dataset
from utils import replace_str_none


class Setup(CTCodeSummarizationMixin, ExperimentSetup):
    def __init__(self, config: DictConfig):
        self.config = config
        config = replace_str_none(config)

        data_transforms_config = config.data_transforms
        self._init_data_transforms(
            max_distance_mask=data_transforms_config.max_distance_mask,
            relative_distances=data_transforms_config.relative_distances,
            distance_binning=data_transforms_config.distance_binning
        )

        dataset_config = config.dataset
        self._init_data(
            language=dataset_config.language,
            use_validation=dataset_config.use_validation,
            use_no_punctuation=dataset_config.use_no_punctuation,
            use_pointer_network=dataset_config.use_pointer_network,
            sort_by_length=dataset_config.sort_by_length,
            chunk_size=dataset_config.chunk_size,
            filter_language=dataset_config.filter_language,
            dataset_imbalance=dataset_config.dataset_imbalance,
            num_sub_tokens=dataset_config.num_sub_tokens,
            num_subtokens_output=dataset_config.num_subtokens_output,
            use_only_ast=dataset_config.use_only_ast,
            mask_all_tokens=dataset_config.mask_all_tokens,
            mini_dataset=False,
            infinite_loading=False,
            max_num_tokens=dataset_config.max_num_tokens
        )
        self.use_pretrained_model = False

        dict_config = OmegaConf.to_container(config.model.encoder)
        config.model.encoder = dict(self.generate_transformer_lm_encoder_config(dict_config))


class CodeTransformerModule(BaseContrastiveDataModule):
    def __init__(
            self,
            config: DictConfig,
            transform: Callable = None
    ):
        super().__init__(
            config=config,
            transform=transform,
        )
        self.setup_ = None

    def prepare_data(self):
        load_dataset(self.config)
        self.setup_ = Setup(self.config)

    def setup(self, stage: str = None):
        assert stage is not None

        stages = []
        if stage == "fit":
            stages += [self.train_holdout, self.val_holdout]
            self.clf_dataset[self.train_holdout] = CodeTransformerDataset(self.setup_.dataset_train)
            self.clf_dataset[self.val_holdout] = CodeTransformerDataset(self.setup_.dataset_validation)
        if stage == "test":
            stages += [self.test_holdout]
            raise ValueError("Not implemented yet")

        if stage == "fit":
            self.contrastive_dataset[self.train_holdout] = ContrastiveDataset(
                clf_dataset=self.clf_dataset[self.train_holdout]
            )

    def collate_single_fn(self, batch: Any) -> Any:
        ctb = self.setup_.dataset_train.collate_fn([sample[0] for sample in batch])
        labels = torch.LongTensor([sample[1] for sample in batch])
        return ctb, labels

    def collate_pair_fn(self, batch: Any) -> Any:
        a_ctb = [sample["a_encoding"] for sample in batch]
        b_ctb = [sample["b_encoding"] for sample in batch]
        a_ctb = self.setup_.dataset_train.collate_fn(a_ctb)
        b_ctb = self.setup_.dataset_train.collate_fn(b_ctb)
        labels = torch.LongTensor([sample["label"] for sample in batch])
        return (a_ctb, b_ctb), labels

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        inputs, label = batch

        if isinstance(inputs, CTBatch):
            inputs = batch_to_device(inputs, device)
        elif isinstance(inputs, Iterable):
            for i, input_ in enumerate(inputs):
                if input_ is not None:
                    inputs[i] = batch_to_device(input_, device)
        else:
            raise ValueError(f"Unsupported type of inputs {type(inputs)}")

        if isinstance(label, torch.Tensor):
            label = label.to(device)
        return inputs, label
