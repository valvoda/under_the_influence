import sys
sys.path.append("../../")

import pytorch_influence_functions as ptif
import torch
from src.preprocess.data_loader import DataPrep
from src.preprocess.test_dataset import TestData
import glob
import os
import argparse
import csv
import shutil


def find_newest(path):
    list_of_files = glob.glob(path)
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def get_score(path, type='val'):
    results_path = path + "/results.csv"
    with open(results_path, 'r') as r_file:
        reader = csv.DictReader(r_file)
        for row in reader:
            return float(row[type+'_loss']), float(row[type+'_f1'])


def find_best(root, input, model):
    results = {}
    best_path = None
    # for dir in os.listdir(root):
        # if not dir.startswith('.') and (dir == 'claims' or dir == 'baseline_positive'):
    best_loss = 100.0
    for experiment in os.listdir(root + "/" + model + "/" + input + "/"):
        if not experiment.startswith('.'):
            path = root + "/" + model + "/" + input + "/" + experiment
            try:
                val_loss, val_f1 = get_score(path)
            except:
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
    parser.add_argument("--model", type=str, default="bert", required=False)
    parser.add_argument("--start", type=int, default=0, required=False)
    parser.add_argument("--end", type=int, default=10000, required=False)
    args = parser.parse_args()

    if args.gpu >= 0:
        device = 'cuda'
    else:
        device = 'cpu'

    sys.path.insert(0, '../train')
    # find the last trained model
    model_path = '../train/trained_models/precedent/'
    model_path = find_best(model_path, args.input, args.model)
    print(f'Best test F1: {get_score(model_path, "test")[1]}, val F1: {get_score(model_path, "val")[1]}')
    # Toy data model test:
    # model_path = '../train/trained_models/precedent/bert/both/e621d7fd7fcb4535a6b48207e1c03dfd'
    print('loaded:', model_path)
    model = torch.load(model_path + '/model.pt', map_location=torch.device(device))

    tokenized_dir = "../datasets/" + 'precedent' + "/" + args.model
    # tokenizer_dir, test, log, max_len, batch_size
    loader = DataPrep(tokenized_dir, args.test, None, args.max_len, args.batch_size, args.input)
    # loader = TestData(args.batch_size, None)
    train_dataloader, val_dataloader, test_dataloader = loader.load(args.start, args.end)

    ptif.init_logging()
    config = ptif.get_default_config()
    config['gpu'] = args.gpu
    config['test_sample_num'] = False
    config['num_classes'] = 14
    # config['recursion_depth'] = 1000
    # config['test_start_index'] = 10
    influences = ptif.calc_img_wise(config, model, train_dataloader, test_dataloader, model_path, args.start, args.end)

    print("DONE!")