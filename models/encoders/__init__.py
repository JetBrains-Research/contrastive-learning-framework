from .code2class import Code2ClassModel
from .gnn import GNNModel
from .transformer import TransformerModel

__all__ = [
    "TransformerModel",
    "Code2ClassModel",
    "GNNModel",
    "encoder_models"
]

encoder_models = {
    "transformer": TransformerModel,
    "code2class": Code2ClassModel,
    "gnn": GNNModel,
}
