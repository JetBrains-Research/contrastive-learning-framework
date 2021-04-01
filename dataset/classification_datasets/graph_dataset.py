import json
from os import listdir
from os.path import join, dirname, isdir

import torch
from omegaconf import DictConfig
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

from utils import is_json_file


class GraphDataset(InMemoryDataset):
    def __init__(self, config: DictConfig, stage: str, transform=None, pre_transform=None):
        self.root = join(config.data_folder, config.dataset.name, config.dataset.dir, stage)
        self.processed_file_path = f"{join(self.root, stage)}.ptg"

        vocab_path = join(config.data_folder, config.dataset.name, config.dataset.dir, config.dataset.vocab_file)
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        self.v_type2id = vocab["v_type2id"]
        self.v_name2id = vocab["v_name2id"]
        self.e_type2id = vocab["e_type2id"]

        super(GraphDataset, self).__init__(self.root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_file_path)

    @property
    def raw_file_names(self):
        raw_files = []
        for class_ in listdir(self.root):
            class_path = join(self.root, class_)
            if isdir(class_path):
                raw_files += [
                    join(class_, file) for file in listdir(class_path) if is_json_file(join(class_path, file))
                ]
        return raw_files

    @property
    def processed_file_names(self):
        return self.processed_file_path

    @property
    def processed_dir(self):
        return self.root

    def process(self):
        data_list = []
        for file in tqdm(self.raw_file_names):
            file_path = join(self.root, file)
            with open(file_path, "r") as f:
                graph = json.load(f)
            e = graph["edges"]
            v = graph["vertexes"]

            x = torch.LongTensor([self.v_type2id[v_["label"]] for v_ in v])
            y = dirname(file)
            edge_index = torch.stack([torch.LongTensor([e_["in"], e_["out"]]) for e_ in e], dim=-1)
            edge_attr = torch.LongTensor([self.e_type2id[e_["label"]] for e_ in e])

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_file_path)

    def get(self, idx):
        graphs = super().get(idx)
        return graphs, graphs.y
