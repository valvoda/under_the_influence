import sys
sys.path.append("../../")

import pytorch_influence_functions as ptif
import torch
from src.preprocess.data_loader import DataPrep
import glob
import os
import argparse
import csv
import shutil


def find_newest(path):
    list_of_files = glob.glob(path)
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def get_validation_score(path):
    results_path = path + "/results.csv"
    with open(results_path, 'r') as r_file:
        reader = csv.DictReader(r_file)
        for row in reader:
            return float(row['val_loss']), float(row['val_f1'])


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
                        val_loss, val_f1 = get_validation_score(path)
                    except:
                        print("Deleted:", path)
                        shutil.rmtree(path)
                        val_loss = 101
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

    if args.gpu >= 0:
        device = 'cuda'
    else:
        device = 'cpu'

    sys.path.insert(0, '../train')
    # find the last trained model
    model_path = '../train/trained_models/precedent/'
    model_path = find_best(model_path, args.input)
    print('loaded:', model_path)
    model = torch.load(model_path + '/model.pt', map_location=torch.device(device))

    tokenized_dir = "../datasets/" + 'precedent' + "/" + 'bert'
    # tokenizer_dir, test, log, max_len, batch_size
    loader = DataPrep(tokenized_dir, args.test, None, args.max_len, args.batch_size, args.input)
    train_dataloader, val_dataloader, test_dataloader = loader.load()

    ptif.init_logging()
    config = ptif.get_default_config()
    config['gpu'] = args.gpu
    config['test_sample_num'] = False
    config['num_classes'] = 14
    # config['test_start_index'] = 2
    influences = ptif.calc_img_wise(config, model, train_dataloader, test_dataloader)

    print("DONE!")