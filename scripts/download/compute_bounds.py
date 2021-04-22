from argparse import ArgumentParser
from collections import defaultdict
from os import listdir
from os.path import join, isdir


def get_task(clones_dir: str):
    parts = clones_dir.split("_")
    return f"{parts[-6]}{parts[-5]}"


def compute_bounds(
        data_path: str,
        train_part: float,
        test_part: float,
):
    sum_ = 0
    task2count = defaultdict(int)
    for clones_dir in listdir(data_path):
        clones_dir_path = join(data_path, clones_dir)
        if isdir(clones_dir_path):
            task2count[get_task(clones_dir)] += 1
            sum_ += 1

    count = 0
    train_bound, test_bound = None, None

    for k in sorted(task2count):
        count += task2count[k]
        if (100 * (count / sum_) > train_part) and (train_bound is None):
            train_bound = count
        if (100 * (count / sum_) > train_part + test_part) and (test_bound is None):
            test_bound = count
    return train_bound, test_bound


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("data_path", type=str)
    arg_parser.add_argument("train_part", type=float, default=60)
    arg_parser.add_argument("test_part", type=float, default=20)
    args = arg_parser.parse_args()

    train_bound_, test_bound_ = compute_bounds(
        data_path=args.data_path,
        train_part=args.train_part,
        test_part=args.test_part,
    )
    print(f"{train_bound_} {test_bound_}")
