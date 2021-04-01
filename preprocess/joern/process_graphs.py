import subprocess
from os import listdir

import pexpect
from os import mkdir
from os.path import exists, join, isfile, dirname, abspath, isdir
from omegaconf import DictConfig
from tqdm import tqdm


def process_graphs(config: DictConfig):
    data_path = join(config.data_folder, config.dataset.name, "raw")
    graphs_path = join(config.data_folder, config.dataset.name, config.dataset.dir)
    cpg_path = join(config.data_folder, config.dataset.name, "cpg")

    if not exists(graphs_path):
        mkdir(graphs_path)

    if not exists(cpg_path):
        mkdir(cpg_path)

    graphs_path = abspath(graphs_path)

    for holdout in [config.train_holdout, config.val_holdout, config.test_holdout]:
        holdout_path = join(data_path, holdout)
        holdout_output = join(graphs_path, holdout)

        if not exists(holdout_output):
            mkdir(holdout_output)

        for class_ in tqdm(listdir(holdout_path)):
            class_path = join(holdout_path, class_)
            if isdir(class_path):
                class_files = [file for file in listdir(class_path) if isfile(join(class_path, file))]
                class_output = join(holdout_output, class_)
                class_cpg = join(cpg_path, class_)

                if not exists(class_output):
                    mkdir(class_output)

                if not exists(class_cpg):
                    mkdir(class_cpg)

                for file in tqdm(class_files):
                    file_path = join(class_path, file)
                    base_name = file.rsplit('.', 1)[0]
                    cpg_file_path = join(class_cpg, f"{base_name}.bin")
                    json_file_name = f"{base_name}.json"
                    json_out = join(class_output, json_file_name)

                    # joern-parse
                    subprocess.check_call([
                        "joern-parse",
                        file_path, "--out",
                        cpg_file_path
                    ])

                    # build graphs
                    subprocess.check_call([
                        "joern",
                        "--script", "preprocess/joern/build_graphs.sc",
                        "--params", f"cpgPath={cpg_file_path},outputPath={json_out}"
                    ])
