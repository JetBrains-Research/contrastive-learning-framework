from os import walk
from os.path import join, splitext


def _in_c_family(file: str):
    _, extension = splitext(file)
    return (extension == ".cpp") or (extension == ".c")


def traverse_clf_dataset(dataset_path: str):
    files = []
    _, base_dirs, _ = next(walk(dataset_path))
    base_dirs_paths = map(lambda file_: join(dataset_path, file_), base_dirs)
    for base_dir_path in base_dirs_paths:
        _, _, dir_files = next(walk(base_dir_path))
        dir_files_paths = map(lambda file_: join(base_dir_path, file_), dir_files)
        dir_files_paths = filter(_in_c_family, dir_files_paths)
        for file in dir_files_paths:
            files.append(file)
    return files
