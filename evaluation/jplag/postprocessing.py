from os.path import join

import regex
from tqdm import tqdm

results_path = "/result"

r = regex.compile(r'Comparing .*.(cpp|c)-.*.(cpp|c): ([0-9]*[.])?[0-9]+')
r_c = regex.compile(r'Comparing .*.c-.*.c: ([0-9]*[.])?[0-9]+')


def postprocess():
    jplag_results_path = join(results_path, "results.txt")
    output_path = join(results_path, "results.csv")
    sim_sep = ": "

    with open(jplag_results_path, 'r') as f:
        total_lines = sum(1 for _ in f)

    with open(jplag_results_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in tqdm(f_in, total=total_lines):
            if r.match(line) is not None:
                if r_c.match(line) is not None:
                    line_ = line[10:].replace(".c-", ".c,")
                else:
                    line_ = line[10:].replace(".cpp-", ".cpp,")
                similarity = float(line_[line_.index(sim_sep) + 2:])
                if similarity:
                    line_ = line_.replace(": ", ",")
                    f_out.write(line_)


if __name__ == "__main__":
    postprocess()
