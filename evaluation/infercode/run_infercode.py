import logging
import pickle
from os import listdir, mkdir
from os.path import join, exists

from infercode.client.infercode_client import InferCodeClient
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

data_path = "/data"


def generate_embeddings():
    dataset_path = join(data_path, "raw", "test")
    storage_path = join(data_path, "infercode")
    if not exists(storage_path):
        mkdir(storage_path)

    infercode = InferCodeClient(language="c")
    infercode.init_from_config()
    vectors = {}
    i = 0

    for cluster in tqdm(listdir(dataset_path), total=len(listdir(dataset_path))):
        if len(list(vectors.keys())) == 1000:
            with open(join(storage_path, f"vectors_{i}.pkl"), "wb") as f:
                pickle.dump(vectors, f)
            vectors = {}
            i += 1
        cluster_path = join(dataset_path, cluster)
        for file in listdir(cluster_path):
            file_path = join(cluster_path, file)
            with open(file_path, "r") as f:
                vectors[file_path] = infercode.encode([f.read()])[0]


if __name__ == "__main__":
    generate_embeddings()
