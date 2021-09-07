from os.path import join

data_dir = "data"


def compute_metrics():
    output_file_path = join(data_dir, "output.txt")
    with open(output_file_path, "r") as f:
        simian_data = f.read()



if __name__ == "__main__":
    compute_metrics()
