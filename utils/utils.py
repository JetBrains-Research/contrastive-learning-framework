from os.path import isfile


def is_json_file(path: str):
    ext = path.rsplit(".", 1)[-1]
    return isfile(path) and (ext == "json")
