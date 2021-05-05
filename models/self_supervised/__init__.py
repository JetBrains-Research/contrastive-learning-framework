from .byol import BYOLModel, BYOLTransform
from .moco import MocoV2Model
from .simclr import SimCLRModel, SimCLRTransform

__all__ = [
    "MocoV2Model",
    "BYOLModel",
    "ssl_models",
    "ssl_models_transforms"
]

ssl_models = {
    "MocoV2": MocoV2Model,
    "BYOL": BYOLModel,
    "SimCLR": SimCLRModel
}

ssl_models_transforms = {
    "BYOL": BYOLTransform,
    "SimCLR": SimCLRTransform
}
