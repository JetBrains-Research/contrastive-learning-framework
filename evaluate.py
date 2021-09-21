import pickle
from argparse import ArgumentParser
from itertools import combinations

import wget
from os import listdir
from os.path import join, dirname, basename, exists

import torch
import yaml
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.utilities.cloud_io import load

from dataset import data_modules
from models import ssl_models, ssl_models_transforms
from models.self_supervised.utils import validation_metrics, compute_map_at_k

data_dir = "data"
checkpoint_storage = "checkpoints"

sweep_path = "https://api.wandb.ai/files/maximzubkov/contrastive-learning-framework"
model2ckpt = {
    "simclr-transformer-poj104": "/i309s5af/epoch=01-val_loss=0.0000-v1.ckpt",
    "moco-transformer-codeforces": "/jkou0zae/epoch=07-val_loss=0.0000.ckpt",
}


def eval_simian(dataset: str):
    if dataset == "poj_104":
        ks = (100, 200, 500)
    elif dataset == "codeforces":
        ks = (5, 10, 15)

    output_file_path = join(data_dir, dataset, "simian.yaml")
    with open(output_file_path, "r") as f:
        simian_output = yaml.safe_load(f)
        simian_output = simian_output["simian"]["checks"][0]["sets"]
    duplicates = [
        {
            "num_duplicated_lines": simian_output[i]["lineCount"],
            "blocks": [elem["sourceFile"] for elem in simian_output[i + 1]["blocks"]]
        }
        for i in range(2, len(simian_output), 2)
    ]
    files = set().union(*[d["blocks"] for d in duplicates])
    file2id = {file: i for i, file in enumerate(files)}
    n_files = len(files)
    duplicate_lines_matrix = np.zeros((n_files, n_files))

    clusters = set().union(*[[basename(dirname(p)) for p in d["blocks"]] for d in duplicates])
    cluster2id = {label: i for i, label in enumerate(clusters)}

    labels = np.zeros(n_files)
    for file in files:
        file_id = file2id[file]
        file_cluster_id = cluster2id[basename(dirname(file))]
        labels[file_id] = file_cluster_id

    for dup in duplicates:
        num_lines = dup["num_duplicated_lines"]
        dup_ids = [file2id[file] for file in dup["blocks"]]
        pairs = combinations(dup_ids, 2)
        for a, b in pairs:
            duplicate_lines_matrix[a, b] += num_lines
            duplicate_lines_matrix[b, a] += num_lines

    duplicate_lines_matrix_ids = np.argsort(duplicate_lines_matrix, axis=-1)
    for k in ks:
        top_ids = duplicate_lines_matrix_ids[:, -k:]
        top_ids = top_ids[:, ::-1].astype(int)

        top_labels = labels[top_ids]
        preds = torch.eq(torch.LongTensor(top_labels), torch.LongTensor(labels).reshape(-1, 1))
        print(f"\tMAP at {k} {round(compute_map_at_k(preds) * 100, 2)}")


def eval_embeddings(model: str, dataset: str):
    storage_path = join(data_dir, dataset, model)
    labels, vectors = [], []
    for file in listdir(storage_path):
        file_path = join(storage_path, file)
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        labels += [basename(dirname(key)) for key in data]
        vectors += [torch.Tensor(data[key]).reshape(1, -1) for key in data]
    labels_set = set(labels)
    label2id = {label: i for i, label in enumerate(labels_set)}
    labels = torch.LongTensor([label2id[label] for label in labels])
    features = torch.cat(vectors, dim=0)
    log = validation_metrics([{"features": features, "labels": labels}], task=dataset)
    for k, v in log.items():
        print(f"\t\t{k}: {v}")


def eval_checkpoint(config_path: str, checkpoint_path: str):
    config = OmegaConf.load(config_path)
    seed_everything(config.seed)

    transform = ssl_models_transforms[config.ssl.name]() if config.ssl.name in ssl_models_transforms else None
    dm = data_modules[config.name](config=config, transform=transform)
    dm.prepare_data()
    dm.setup("test")

    model = ssl_models[config.ssl.name](config=config)
    checkpoint = load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    gpu = -1 if torch.cuda.is_available() else None
    trainer = Trainer(gpus=gpu)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--model", type=str)
    arg_parser.add_argument("--dataset", type=str)
    arg_parser.add_argument("--checkpoint_path", type=str, default=None)
    arg_parser.add_argument("--config_path", type=str, default=None)
    args = arg_parser.parse_args()

    if args.model in ["infercode", "transcoder-1", "transcoder-2"]:
        eval_embeddings(args.model, args.dataset)
    elif args.model == "simian":
        eval_simian(args.dataset)
    elif f"{args.model}-{args.dataset}" in model2ckpt:
        if (args.checkpoint_path is not None) and (args.config_path is not None):
            config_path = args.config_path
            checkpoint_path = args.checkpoint_path
        else:
            full_name = f"{args.model}-{args.dataset}"
            config_path = join("configs", f"{full_name}.yaml")
            checkpoint_path = join(checkpoint_storage, f"{full_name}.ckpt")
            if not exists(checkpoint_path):
                wget.download(sweep_path + model2ckpt[full_name], checkpoint_path)
        eval_checkpoint(config_path=config_path, checkpoint_path=checkpoint_path)
    else:
        raise ValueError(f"Unknown model {args.model}")
