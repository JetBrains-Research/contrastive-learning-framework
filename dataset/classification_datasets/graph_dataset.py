import json
from os import listdir
from os.path import join, dirname

import torch
from omegaconf import DictConfig
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm


class GraphyDataset(InMemoryDataset):
    def __init__(self, config: DictConfig, split: str, transform=None, pre_transform=None):
        root = join(config.data_folder, config.dataset.name, config.dataset.dir)
        super(GraphyDataset, self).__init__(root, transform, pre_transform)

        self.data_dir = join(root, split)
        self.processed_file_path = f"{join(root, split)}.ptg"
        self.data, self.slices = torch.load(self.processed_paths)

        vocab_path = join(root, config.dataset.vocab_file)
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        self.v_type2id = vocab["v_type2id"]
        self.v_name2id = vocab["v_name2id"]
        self.e_type2id = vocab["e_type2id"]

    @property
    def raw_file_names(self):
        raw_files = []
        for class_ in listdir(self.data_dir):
            class_path = join(self.data_dir, class_)
            raw_files += [join(class_, file) for file in listdir(class_path)]
        return raw_files

    @property
    def processed_file_names(self):
        return self.processed_file_path

    def download(self):
        pass

    def process(self):
        data_list = []
        for file in tqdm(self.raw_paths):
            with open(file, "r") as f:
                graph = json.load(f)
            e = graph["edges"]
            v = graph["vertexes"]

            x = torch.LongTensor([self.v_type2id[v_["label"]] for v_ in v]).unsqueeze(1)
            y = dirname(file)
            edge_index = torch.stack([torch.LongTensor([e_["in"], e_["out"]]) for e_ in e])
            edge_attr = torch.Tensor([self.e_type2id[e_["label"]] for e_ in e]).unsqueeze(1)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths)
