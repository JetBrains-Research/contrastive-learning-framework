from code_transformer.configuration.transformer_lm_encoder import TransformerLMEncoderConfig
from code_transformer.modeling.code_transformer.lm import TransformerLMEncoder
from code_transformer.preprocessing.datamanager.base import CTBatch
from omegaconf import DictConfig
from torch import nn


class CodeTransformerModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.num_classes = config.num_classes
        self.code_transformer = TransformerLMEncoder(TransformerLMEncoderConfig(**config.model.encoder))

    def forward(self, batch: CTBatch, need_weights: bool = False):
        transformer_output = self.code_transformer.forward(
            input_tokens=batch.tokens,
            input_node_types=batch.node_types,
            input_token_types=batch.token_types,
            relative_distances=batch.relative_distances,
            attention_mask=batch.perm_mask,
            pad_mask=1 - batch.pad_mask,
            target_mapping=None,
            need_weights=need_weights,
            languages=batch.languages
        )
        return transformer_output.out_emb[:, 0, :]
