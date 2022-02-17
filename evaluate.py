import json
import pickle
from argparse import ArgumentParser
from itertools import combinations, chain
from os import listdir
from os.path import join, dirname, basename, exists

import numpy as np
import torch
import wget
import yaml
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
    "moco-transformer-poj104": "/3gbb9ude/epoch=02-val_loss=0.0000-v1.ckpt",
    "swav-transformer-poj104": "/1pqdjf50/epoch=00-val_loss=0.0000.ckpt",
    "simclr-transformer-codeforces": "/z3n7pcg4/epoch=08-val_loss=0.0000.ckpt",
    "moco-transformer-codeforces": "/jkou0zae/epoch=06-val_loss=0.0000.ckpt",
    "swav-transformer-codeforces": "/lgk13t5u/epoch=01-val_loss=0.0000.ckpt",

    "simclr-code2class-poj104": "/b4qs2uam/epoch=00-val_loss=0.0000.ckpt",
    "moco-code2class-poj104": "/rrvjo40p/epoch=00-val_loss=0.0000-v1.ckpt",
    "swav-code2class-poj104": "/smyuztja/epoch=00-val_loss=0.0000-v1.ckpt",
    "simclr-code2class-codeforces": "/3bnriznw/epoch=04-val_loss=0.0000.ckpt",
    "moco-code2class-codeforces": "/vvs854oi/epoch=07-val_loss=0.0000.ckpt",
    "swav-code2class-codeforces": "/jkkxhxsw/epoch=07-val_loss=0.0000.ckpt",

    "simclr-gnn-poj104": "/56q7ksbb/epoch=00-val_loss=0.0000-v1.ckpt",
    "moco-gnn-poj104": "/tgj7vpyi/epoch=00-val_loss=0.0000-v1.ckpt",
    "swav-gnn-poj104": "/lwdxtex6/epoch=00-val_loss=0.0000-v1.ckpt",
    "simclr-gnn-codeforces": "/uu1ries9/epoch=07-val_loss=0.0000.ckpt",
    "moco-gnn-codeforces": "/q89d3j8h/epoch=07-val_loss=0.0000.ckpt",
    "swav-gnn-codeforces": "/5wopmvq8/epoch=07-val_loss=0.0000.ckpt",
}


def collect_labels(dataset: str):
    test_set_path = join(data_dir, dataset, "raw", "val_tmp")
    files_total = list(chain(*[
        [join(d, f) for f in listdir(join(test_set_path, d))]
        for d in listdir(test_set_path)
    ]))
    file2id = {file: i for i, file in enumerate(files_total)}
    n_total_files = len(files_total)

    clusters = set().union([basename(dirname(f)) for f in files_total])
    cluster2id = {label: i for i, label in enumerate(clusters)}

    labels = np.zeros(n_total_files)
    for file in files_total:
        file_id = file2id[file]
        labels[file_id] = cluster2id[basename(dirname(file))]

    return labels, file2id, cluster2id


def build_jplag_matrix(dataset: str, file2id: dict):
    output_file_path = join(data_dir, dataset, "jplag", "results.csv")
    duplicate_lines_matrix = np.zeros((len(file2id), len(file2id)))
    with open(output_file_path, "r") as f:
        for line in f:
            file1, file2, score = line.split(";")[1:-1]
            file1 = join(file1.rsplit("_", 2)[0], "_".join(file1.rsplit("_", 2)[1:]))
            file2 = join(file2.rsplit("_", 2)[0], "_".join(file2.rsplit("_", 2)[1:]))
            file1_id, file2_id = file2id[file1], file2id[file2]
            duplicate_lines_matrix[file1_id, file2_id] += float(score)
            duplicate_lines_matrix[file2_id, file1_id] += float(score)
    return duplicate_lines_matrix


def build_simian_matrix(dataset: str, file2id: dict):
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

    duplicate_lines_matrix = np.zeros((len(file2id), len(file2id)))
    for dup in duplicates:
        num_lines = dup["num_duplicated_lines"]
        dup_ids = [file2id[join(basename(dirname(file)), basename(file))] for file in dup["blocks"]]
        pairs = combinations(dup_ids, 2)
        for a, b in pairs:
            duplicate_lines_matrix[a, b] += num_lines
            duplicate_lines_matrix[b, a] += num_lines

    return duplicate_lines_matrix


def eval_from_matrix(dataset: str, model: str):
    if dataset == "poj_104":
        ks = (100, 200, 500)
    elif dataset == "codeforces":
        ks = (5, 10, 15)

    labels, file2id, cluster2id = collect_labels(dataset)

    if model == "simian":
        duplicate_lines_matrix = build_simian_matrix(dataset, file2id)
    elif model == "jplag":
        duplicate_lines_matrix = build_jplag_matrix(dataset, file2id)
    else:
        raise ValueError(f"Unknown model {model}")

    zero_lines = (duplicate_lines_matrix.sum(-1) == 0)
    duplicate_lines_matrix_ids = np.argsort(duplicate_lines_matrix, axis=-1)
    for k in ks:
        top_ids = duplicate_lines_matrix_ids[:, -k:]
        top_ids = top_ids[:, ::-1].astype(int)

        top_labels = labels[top_ids]
        top_labels[zero_lines, :] = -1
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
        vectors += [torch.Tensor(data[key].cpu()).reshape(1, -1) for key in data]
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

    model = ssl_models[config.ssl.name](config=config)
    checkpoint = load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    gpu = -1 if torch.cuda.is_available() else None
    trainer = Trainer(gpus=gpu)
    res = trainer.test(model, datamodule=dm)
    return res


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--model", type=str)
    arg_parser.add_argument("--dataset", type=str)
    arg_parser.add_argument("--checkpoint_path", type=str, default=None)
    arg_parser.add_argument("--config_path", type=str, default=None)
    args = arg_parser.parse_args()

    if args.model in ["infercode", "transcoder-1", "transcoder-2"]:
        eval_embeddings(args.model, args.dataset)
    elif args.model in ["simian", "jplag"]:
        eval_from_matrix(args.dataset, args.model)
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
        res = eval_checkpoint(config_path=config_path, checkpoint_path=checkpoint_path)
        with open(join(data_dir, f"{args.model}-{args.dataset}.json"), 'w') as f:
            json.dump(res, f)
    else:
        raise ValueError(f"Unknown model {args.model}")
