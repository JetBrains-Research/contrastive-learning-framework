from .code2class import Code2ClassModel
from .gnn import GNNModel
from .siamese_arm import SiameseArm
from .transformer import TransformerModel
from .code_transformer import CodeTransformerModel

__all__ = [
    "TransformerModel",
    "Code2ClassModel",
    "CodeTransformerModel",
    "GNNModel",
    "SiameseArm",
    "encoder_models"
]

encoder_models = {
    "transformer": TransformerModel,
    "code2class": Code2ClassModel,
    "gnn": GNNModel,
    "code-transformer": CodeTransformerModel
}
