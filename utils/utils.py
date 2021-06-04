from os.path import isfile


def is_json_file(path: str):
    ext = path.rsplit(".", 1)[-1]
    return isfile(path) and (ext == "json")


def get_task(label: str):
    parts = label.split("_")
    return f"{parts[-6]}{parts[-5]}"
