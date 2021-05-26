import json
from os.path import join

import torch.nn as nn
from omegaconf import DictConfig
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import GENConv


class GNNModel(nn.Module):
    def __init__(self, config: DictConfig):
        super(GNNModel, self).__init__()

        self.num_classes = config.num_classes

        vocab_path = join(config.data_folder, config.dataset.name, config.dataset.dir, config.dataset.vocab_file)
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        num_v_types = len(vocab["v_type2id"].keys())
        num_e_types = len(vocab["e_type2id"].keys())

        self.gen_conv = GENConv(
            in_channels=config.encoder.embedding_size,
            out_channels=config.encoder.out_channels,
            num_layers=config.encoder.num_layers
        )

        self.edge_type_embedding = nn.Embedding(
            num_embeddings=num_e_types,
            embedding_dim=config.encoder.embedding_size
        )
        self.vertex_type_embedding = nn.Embedding(
            num_embeddings=num_v_types,
            embedding_dim=config.encoder.embedding_size
        )
        self.fc1 = nn.Linear(
            in_features=config.encoder.out_channels,
            out_features=config.encoder.hidden_size
        )

        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(
            in_features=config.encoder.hidden_size,
            out_features=config.num_classes
        )

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x_embed = self.vertex_type_embedding(x)
        edge_weight_embed = self.edge_type_embedding(edge_weight)
        out = self.gen_conv(x_embed, edge_index, edge_weight_embed)
        out = global_mean_pool(out, data.batch)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out
