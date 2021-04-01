from os.path import isfile

c_family_exts = ["c", "cpp"]


def is_c_family_file(path: str):
    ext = path.rsplit(".", 1)[-1]
    return isfile(path) and (ext in c_family_exts)
