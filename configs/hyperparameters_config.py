from dataclasses import dataclass


@dataclass(frozen=True)
class ModelHyperparameters:
    n_epochs: int
    batch_size: int
    val_check_interval: float
    log_every_n_steps: int = 200


default_hyperparametrs = ModelHyperparameters(
    n_epochs=10,
    batch_size=16,
    val_check_interval=0.1,
)