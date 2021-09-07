from itertools import combinations
from os.path import join, dirname, basename

import numpy as np
import torch
import yaml

from models.self_supervised.utils import compute_map_at_k

data_dir = "data"


def compute_metrics(dataset, ks):
    output_file_path = join(data_dir, dataset, "output.yaml")
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


if __name__ == "__main__":
    print("Codeforces:")
    compute_metrics("codeforces", ks=(5, 10, 15))
    print("POJ 104:")
    compute_metrics("poj_104", ks=(100, 200, 500))
