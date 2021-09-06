import subprocess
import sys
from argparse import ArgumentParser
from os import listdir, mkdir
from os.path import join, exists, isfile
from random import seed
from shutil import move

from omegaconf import DictConfig, OmegaConf

from preprocess import tokenize, process_graphs, build_graphs_vocab, build_code2seq_vocab, process_astminer_csv

DOWNLOAD_SCRIPT = "download_data.sh"


def load_dataset(config: DictConfig):
    if not exists(config.data_folder):
        mkdir(config.data_folder)
    seed_ = config.seed
    dataset_path = join(config.data_folder, config.dataset.name)
    if not exists(dataset_path):
        subprocess.run(
            args=[
                "bash",
                join("scripts", "download", "download_data.sh"),
                "--dataset", config.dataset.name,
            ],
            stderr=sys.stderr,
            stdout=sys.stdout
        )

    seed(seed_)
    if config.name == "code2class":
        storage_path = join(dataset_path, config.dataset.dir)
        if not exists(storage_path):
            subprocess.run(
                args=[
                    "bash",
                    join("scripts", "run_astminer.sh"),
                    "--dataset", config.dataset.name,
                ],
                stderr=sys.stderr,
                stdout=sys.stdout
            )
            for holdout in [config.train_holdout, config.val_holdout, config.test_holdout]:
                print(f"preprocessing {holdout} data")
                process_astminer_csv(
                    data_folder=config.data_folder,
                    dataset_name=config.dataset.name,
                    holdout_name=holdout,
                    is_shuffled=True
                )

            mkdir(storage_path)
            for elem in listdir(dataset_path):
                path_ = join(dataset_path, elem)
                if isfile(path_):
                    if elem.rsplit(".", 1)[1] in ["csv", "c2s"]:
                        move(path_, storage_path)
        if not exists(join(storage_path, "vocabulary.pkl")):
            build_code2seq_vocab(*[
                join(storage_path, f"{config.dataset.name}.{holdout}.c2s") for holdout in [
                    config.train_holdout,
                    config.test_holdout,
                    config.val_holdout
                ]
            ])

    elif config.name == "transformer":
        storage_path = join(dataset_path, config.dataset.dir)
        if not exists(join(storage_path, config.dataset.tokenizer_name)):
            tokenize(config)
    elif config.name == "gnn":
        storage_path = join(dataset_path, config.dataset.dir)
        holdouts_existence = [
            exists(join(storage_path, holdout)) for holdout in [
                config.train_holdout,
                config.val_holdout,
                config.test_holdout
            ]
        ]
        if not all(holdouts_existence):
            process_graphs(config)

        build_graphs_vocab(config)
    elif config.name == "code-transformer":
        if not (exists(join(dataset_path, config.dataset.dir1)) and exists(join(dataset_path, config.dataset.dir2))):
            script = ["python", "-m", "scripts.run-preprocessing"]
            for holdout in [config.train_holdout, config.val_holdout, config.test_holdout]:
                for stage in [1, 2]:
                    stage_script = script + [
                        join(
                            "configs",
                            "code-transformer",
                            f"preprocess-{stage}.yaml"
                        ),
                        config.dataset.name,
                        holdout
                    ]
                    subprocess.call(' '.join(stage_script), shell=True)
    else:
        raise ValueError(f"Model {config.name} is not currently supported")


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config_path", type=str)
    args = arg_parser.parse_args()

    config_ = OmegaConf.load(args.config_path)
    load_dataset(config_)
