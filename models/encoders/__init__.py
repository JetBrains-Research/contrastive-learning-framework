from .code2class import Code2ClassModel
from .lstm import LSTMModel

__all__ = [
    "LSTMModel",
    "Code2ClassModel",
    "encoder_models"
]

encoder_models = {
    "lstm": LSTMModel,
    "code2class": Code2ClassModel
}
