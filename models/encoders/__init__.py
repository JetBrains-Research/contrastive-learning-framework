from .code2class import Code2ClassModel
from .gnn import GNNModel
from .transformer import TransformerModel
from .siamese_arm import SiameseArm

__all__ = [
    "TransformerModel",
    "Code2ClassModel",
    "GNNModel",
    "SiameseArm",
    "encoder_models"
]

encoder_models = {
    "transformer": TransformerModel,
    "code2class": Code2ClassModel,
    "gnn": GNNModel,
}
