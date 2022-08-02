import sys
sys.path.append("../../")

import os
import argparse
import csv
import shutil


def get_score(path, type='val'):
    results_path = path + "/results.csv"
    with open(results_path, 'r') as r_file:
        reader = csv.DictReader(r_file)
        for row in reader:
            return float(row[type+'_loss']), float(row[type+'_f1'])


def find_best(root, input):
    results = {}
    best_path = None
    for dir in os.listdir(root):
        # if not dir.startswith('.') and (dir == 'claims' or dir == 'baseline_positive'):
            best_loss = 100.0
            for experiment in os.listdir(root + dir + "/" + input + "/"):
                if not experiment.startswith('.'):
                    path = root + dir + "/" + input + "/" + experiment
                    try:
                        val_loss, val_f1 = get_score(path)
                    except:
                        pass
                    if val_loss < best_loss:
                        # print(path, val_f1)
                        best_loss = val_loss
                        best_path = path

    return best_path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=512, required=False)
    parser.add_argument("--gpu", type=int, default=1, required=False)
    parser.add_argument("--batch_size", type=int, default=8, required=False)
    parser.add_argument("--test", dest='test', action='store_true')
    parser.add_argument("--input", type=str, default="facts", required=False)  # arguments
    args = parser.parse_args()

    path = '../train/trained_models/precedent/'
    best_path = find_best(path, args.input)
    print(best_path)
    print(f'Best test F1: {get_score(best_path, "test")[1]}, val F1: {get_score(best_path, "val")[1]}')
