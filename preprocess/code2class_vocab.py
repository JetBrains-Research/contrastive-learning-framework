from collections import Counter
from os.path import join, exists
from typing import Counter as TypeCounter, Any
from typing import Dict

from code2seq.preprocessing.build_vocabulary import (
    count_lines_in_file,
    parse_token,
    parse_path_context,
    _counters_to_vocab,
    convert_vocabulary
)
from code2seq.utils.vocabulary import Vocabulary
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def collect_vocabulary(config: Dict[str, Any], train_holdout: str, test_holdout: str, val_holdout: str) -> Vocabulary:
    counters: Dict[str, TypeCounter[str]] = {k: Counter() for k in ["target", "token", "path", "type"] if k in config}
    with open(train_holdout, "r") as train_file:
        for line in tqdm(train_file, total=count_lines_in_file(train_holdout)):
            label, *path_contexts = line.split()
            counters["target"].update(parse_token(label, config["target"]["is_splitted"]))
            for path_context in path_contexts:
                parsed_context = parse_path_context(config, path_context.split(","))
                for key, value in parsed_context.items():
                    counters[key].update(value)
    with open(test_holdout, "r") as test_file:
        for line in tqdm(test_file, total=count_lines_in_file(test_holdout)):
            label, *_ = line.split()
            counters["target"].update(parse_token(label, config["target"]["is_splitted"]))
    with open(val_holdout, "r") as val_file:
        for line in tqdm(val_file, total=count_lines_in_file(val_holdout)):
            label, *_ = line.split()
            counters["target"].update(parse_token(label, config["target"]["is_splitted"]))
    return _counters_to_vocab(config, counters)


def build_code2seq_vocab(config: DictConfig):
    dataset_directory = join(config.data_folder, config.dataset.name)
    possible_dict = join(dataset_directory, f"{config.dataset.name}.dict.c2s")

    train_holdout = join(dataset_directory, f"{config.dataset.name}.{config.train_holdout}.c2s")
    test_holdout = join(dataset_directory, f"{config.dataset.name}.{config.test_holdout}.c2s")
    val_holdout = join(dataset_directory, f"{config.dataset.name}.{config.val_holdout}.c2s")

    dict_data_config = OmegaConf.to_container(config.dataset, True)
    if not isinstance(dict_data_config, dict):
        raise ValueError
    if exists(possible_dict):
        vocabulary = convert_vocabulary(dict_data_config, possible_dict)
    else:
        vocabulary = collect_vocabulary(dict_data_config, train_holdout, test_holdout, val_holdout)
    vocabulary.dump_vocabulary(join(dataset_directory, "vocabulary.pkl"))
