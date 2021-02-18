from argparse import ArgumentParser
from os import listdir, remove
from os.path import splitext, join, isdir, exists
from dataclasses import asdict

import youtokentome as yttm

from configs import default_tokenizer_config

data = "data"


def tokenize(dataset_path: str, model_name: str = "model.yttm"):
    model_path = join(dataset_path, model_name)
    buffer_path = "text.yttm"
    if exists(buffer_path):
        remove(buffer_path)

    for file in listdir(dataset_path):
        transformed_files_path = join(dataset_path, file)
        if isdir(transformed_files_path):
            for transformed_file in listdir(transformed_files_path):
                file_path = join(transformed_files_path, transformed_file)
                _, ext = splitext(file_path)
                if ext == ".cpp":
                    with open(file_path, "r", encoding="utf8", errors='ignore') as file_:
                        text = file_.read() + "\n"
                        with open(buffer_path, "a") as buffer_:
                            buffer_.write(text)

    tokenizer_config = default_tokenizer_config
    _ = yttm.BPE.train(
        data="text.yttm",
        model=model_path,
        **asdict(tokenizer_config)
    )

    remove("text.yttm")


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="poj_104")
    arg_parser.add_argument("--model_name", type=str, default="model.yttm")
    arg_parser.add_argument("--is_test", action="store_true")
    args = arg_parser.parse_args()
    dataset_path_ = join(data, args.dataset)
    model_path_ = join(dataset_path_, args.model_name)
    tokenize(dataset_path=dataset_path_, model_path=model_path_)
