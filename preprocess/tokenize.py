from argparse import ArgumentParser
from os import listdir, remove
from os.path import splitext, join, isdir

import youtokentome as yttm

data = "data"


def _tokenize(dataset_path: str, model_path: str):
    text = ""
    for file in listdir(dataset_path):
        transformed_files_path = join(dataset_path, file)
        if isdir(transformed_files_path):
            for transformed_file in listdir(transformed_files_path):
                file_path = join(transformed_files_path, transformed_file)
                _, ext = splitext(file_path)
                if ext == ".cpp":
                    with open(file_path, "r", encoding="utf8", errors='ignore') as f:
                        text += f.read() + "\n"
    with open("text.yttm", "w") as f:
        f.write(text)

    _ = yttm.BPE.train(
        data="text.yttm",
        model=model_path,
        vocab_size=30000
    )

    remove("text.yttm")


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="poj_104")
    arg_parser.add_argument("--model_name", type=str, default="model.yttm")
    args = arg_parser.parse_args()
    dataset_path = join(data, args.dataset)
    model_path = join(dataset_path, args.model_name)
    _tokenize(dataset_path=dataset_path, model_path=model_path)
