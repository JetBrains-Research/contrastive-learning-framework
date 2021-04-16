import subprocess
import sys
from argparse import ArgumentParser
from os import listdir, mkdir
from os.path import join, exists, isfile
from random import seed
from shutil import move

from code2seq.preprocessing.astminer_to_code2seq import preprocess_csv
from code2seq.preprocessing.build_vocabulary import preprocess as build_code2seq_vocab
from omegaconf import DictConfig, OmegaConf

from preprocess import tokenize, process_graphs, build_graphs_vocab

TRAIN_PART = 0.7
VAL_PART = 0.2
TEST_PART = 0.1
DOWNLOAD_SCRIPT = "download_data.sh"


def load_poj_104(config: DictConfig):
    seed_ = config.seed
    dataset_path = join(config.data_folder, config.dataset.name)
    if not exists(dataset_path):
        subprocess.run(
            args=[
                "sh",
                join("scripts", "download", "download_data.sh"),
                "--dataset", config.dataset.name,
                "--dev",
                "--astminer", join("build", "astminer")
            ],
            stderr=sys.stderr,
            stdout=sys.stdout
        )

    seed(seed_)
    if config.name == "code2class":
        if not exists(join(dataset_path, config.dataset.dir)):
            for holdout in [config.train_holdout, config.val_holdout, config.test_holdout]:
                print(f"preprocessing {holdout} data")
                preprocess_csv(
                    data_folder=config.data_folder,
                    dataset_name=config.dataset.name,
                    holdout_name=holdout,
                    is_shuffled=config.hyper_parameters.shuffle_data
                )
            build_code2seq_vocab(config)

            dataset_path = join(config.data_folder, config.dataset.name)
            paths_storage = join(dataset_path, config.dataset.dir)
            mkdir(paths_storage)
            for elem in listdir(dataset_path):
                path_ = join(dataset_path, elem)
                if isfile(path_):
                    if (elem.rsplit(".", 1)[1] in ["csv", "c2s"]) or (elem == "vocabulary.pkl"):
                        move(path_, paths_storage)

    elif config.name == "transformer":
        if not exists(join(dataset_path, config.dataset.dir, config.dataset.tokenizer_name)):
            tokenize(config)
    elif config.name == "gnn":
        holdouts_existence = [
            exists(join(dataset_path, config.dataset.dir, holdout)) for holdout in [
                config.train_holdout,
                config.val_holdout,
                config.test_holdout
            ]
        ]
        if not all(holdouts_existence):
            process_graphs(config)

        build_graphs_vocab(config)
    else:
        raise ValueError(f"Model {config.name} is not currently supported")


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config_path", type=str)
    args = arg_parser.parse_args()

    config_ = OmegaConf.load(args.config_path)
    load_poj_104(config_)
