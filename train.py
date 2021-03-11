from argparse import ArgumentParser

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from configs import default_config, test_config, default_hyperparametrs
from dataset import BaseDataModule
from models import encoder_models, ssl_models, ssl_models_transforms

SEED = 9


def train(model: str, encoder: str, dataset: str, is_test: bool, log_offline: bool, resume: str = None):
    seed_everything(SEED)

    if encoder not in encoder_models:
        print(f"Unknown encoder: {encoder}, try on of {encoder_models}")

    if model not in ssl_models:
        print(f"Unknown model: {model}, try on of {ssl_models.keys()}")

    config = test_config if is_test else default_config
    hyperparams = default_hyperparametrs

    transform = ssl_models_transforms[model]()
    dm = BaseDataModule(dataset, is_test=is_test, batch_size=hyperparams.batch_size, transform=transform)

    model_ = ssl_models[model](
        base_encoder=encoder,
        encoder_config=config,
        batch_size=hyperparams.batch_size,
        max_epochs=hyperparams.n_epochs,
        num_classes=dm.num_classes
    )
    # define logger
    wandb_logger = WandbLogger(
        project=f"{model}-{encoder}-{dataset}",
        log_model=True,
        offline=log_offline
    )
    wandb_logger.watch(model_)
    # define model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir,
        filename="{epoch:02d}-{val_loss:.4f}",
        period=3,
        save_top_k=-1,
    )
    # use gpu if it exists
    gpu = 1 if torch.cuda.is_available() else None
    # define learning rate logger
    lr_logger = LearningRateMonitor("step")
    trainer = Trainer(
        max_epochs=hyperparams.n_epochs,
        val_check_interval=hyperparams.val_check_interval,
        log_every_n_steps=hyperparams.log_every_n_steps,
        logger=wandb_logger,
        gpus=gpu,
        callbacks=[lr_logger, checkpoint_callback],
        resume_from_checkpoint=resume,
    )

    trainer.fit(model=model_, datamodule=dm)
    trainer.test()


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("model", type=str, default="MocoV2", choices=list(ssl_models.keys()))
    arg_parser.add_argument("encoder", type=str, default="LSTM", choices=list(encoder_models))
    arg_parser.add_argument("--dataset", type=str, default=None)
    arg_parser.add_argument("--offline", action="store_true")
    arg_parser.add_argument("--is_test", action="store_true")
    arg_parser.add_argument("--resume", type=str, default=None)
    args = arg_parser.parse_args()

    train(
        model=args.model,
        encoder=args.encoder,
        dataset=args.dataset,
        log_offline=args.offline,
        is_test=args.is_test,
        resume=args.resume
    )
