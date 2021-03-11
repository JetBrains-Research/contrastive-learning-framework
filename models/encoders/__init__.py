from .lstm import LSTMModel
from code2seq.model import Code2Class

__all__ = [
    "LSTMModel",
    "encoder_models"
]

encoder_models = {
    "LSTM": LSTMModel,
    "Code2Class": Code2Class
}
