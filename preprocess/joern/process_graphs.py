from os import listdir

import pexpect
import os
from os.path import exists, join, isfile, dirname, abspath
from omegaconf import DictConfig
from tqdm import tqdm


def process_graphs(config: DictConfig):
    joern_process = pexpect.spawnu("sh", ["./" + config.joern_path + "joern"])

    data_path = join(config.data_folder, config.dataset.name)
    for holdout in [config.train_holdout, config.val_holdout, config.test_holdout]:
        holdout_path = join(data_path, holdout)
        holdout_files = [file for file in listdir(holdout_path) if isfile(join(holdout_path, file))]
        holdout_output = abspath(join(holdout_path, config.graphs_dir))

        for file in tqdm(holdout_files):
            file_path = join(holdout_path, file)
            json_file_name = f"{file.rsplit('.')[0]}.json"

            json_out = join(holdout_output, json_file_name)
            import_cpg_cmd = f"importCpg(\"{file_path}\")"
            script_path = join(abspath(config.script_path), "graph-for-funcs.sc")
            run_script_cmd = f"cpg.runScript(\"{script_path}\").toString() |> \"{json_out}\" "
            delete_cmd = "delete"

            for cmd in [import_cpg_cmd, run_script_cmd, delete_cmd]:
                joern_process.expect('joern>', searchwindowsize=50)
                joern_process.sendline(cmd)
                joern_process.buffer = ""
