from .byol import BYOLModel, BYOLTransform
from .moco import MocoV2Model

__all__ = [
    "MocoV2Model",
    "BYOLModel",
    "ssl_models",
    "ssl_models_transforms"
]

ssl_models = {
    "MocoV2": MocoV2Model,
    "BYOL": BYOLModel
}

ssl_models_transforms = {
    "MocoV2": None,
    "BYOL": BYOLTransform
}
