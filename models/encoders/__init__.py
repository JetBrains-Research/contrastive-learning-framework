from .code2class import Code2ClassModel
from .code_transformer import CodeTransformerModel
from .gnn import GNNModel
from .siamese_arm import SiameseArm
from .transformer import TransformerModel

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
