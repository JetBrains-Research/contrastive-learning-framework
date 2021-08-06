from code_transformer.configuration.code_transformer import CodeTransformerCoreConfig
from code_transformer.modeling.code_transformer.code_transformer import CodeTransformer
from omegaconf import DictConfig
from torch import nn


class CodeTransformerModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.num_classes = config.num_classes
        self.code_transformer = CodeTransformer(CodeTransformerCoreConfig(**config.nodel.encoder.transformer))

    def forward(self, *args, **kwargs):
        return self.code_transformer.forward_batch(*args, **kwargs)
