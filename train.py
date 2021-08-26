from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from dataset import data_modules
from models import encoder_models, ssl_models, ssl_models_transforms

SEED = 9


def train(config_path: str, resume: str, seed: int = None, learning_rate: float = None):
    config = OmegaConf.load(config_path)

    seed_everything(seed if seed is not None else config.seed)

    if learning_rate is not None:
        config.ssl.learning_rate = learning_rate

    encoder = config.name
    ssl_model = config.ssl.name
    dataset = config.dataset

    if encoder not in encoder_models:
        print(f"Unknown encoder: {encoder}, try on of {encoder_models}")

    if ssl_model not in ssl_models:
        print(f"Unknown model: {ssl_model}, try on of {ssl_models.keys()}")

    transform = ssl_models_transforms[ssl_model]() if ssl_model in ssl_models_transforms else None
    dm = data_modules[encoder](
        config=config,
        transform=transform
    )

    dm.prepare_data()

    model_ = ssl_models[ssl_model](
        config=config,
    )

    # define logger
    wandb_logger = WandbLogger(
        project=f"{ssl_model}-{encoder}-{dataset.name}",
        log_model=True,
        offline=config.log_offline
    )
    wandb_logger.watch(model_)
    # define model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir,
        filename="{epoch:02d}-{val_loss:.4f}",
        period=config.save_every_epoch,
        save_top_k=-1,
    )
    # use gpu if it exists
    gpu = -1 if torch.cuda.is_available() else None
    # define learning rate logger
    lr_logger = LearningRateMonitor("step")
    trainer = Trainer(
        max_epochs=config.hyper_parameters.n_epochs,
        val_check_interval=config.val_check_interval,
        log_every_n_steps=config.log_every_n_steps,
        logger=wandb_logger,
        gpus=gpu,
        callbacks=[lr_logger, checkpoint_callback],
        resume_from_checkpoint=resume,
    )

    trainer.fit(model=model_, datamodule=dm)
    trainer.test()


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config_path", type=str)
    arg_parser.add_argument("--resume", type=str, default=None)
    arg_parser.add_argument("--seed", type=int, default=None)
    arg_parser.add_argument("--learning_rate", type=float, default=None)
    args = arg_parser.parse_args()

    train(
        config_path=args.config_path,
        resume=args.resume,
        seed=args.seed,
        learning_rate=args.learning_rate
    )
