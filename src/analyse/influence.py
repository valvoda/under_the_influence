import sys
sys.path.append("../../")

import pytorch_influence_functions as ptif
import torch
from src.train.data_loader import DataPrep
import pickle
import json
import glob
import os
import argparse
import numpy as np

def find_newest(path):
    list_of_files = glob.glob(path)
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=512, required=False)
    parser.add_argument("--gpu", type=int, default=1, required=False)
    parser.add_argument("--batch_size", type=int, default=8, required=False)
    parser.add_argument("--test", dest='test', action='store_true')
    args = parser.parse_args()

    if args.gpu >= 0:
        device = 'cuda'
    else:
        device = 'cpu'

    sys.path.insert(0, '../train')
    # find the last trained model
    model_path = '../train/trained_models/precedent/bert/facts/*'
    model_path = find_newest(model_path)
    print('loaded:', model_path)
    model = torch.load(model_path + '/model.pt', map_location=torch.device(device))

    tokenized_dir = "../datasets/" + 'precedent' + "/" + 'bert'
    # tokenizer_dir, test, log, max_len, batch_size
    loader = DataPrep(tokenized_dir, args.test, None, args.max_len, args.batch_size)
    train_dataloader, val_dataloader, test_dataloader = loader.load()

    ptif.init_logging()
    config = ptif.get_default_config()
    config['gpu'] = args.gpu
    config['test_sample_num'] = False
    config['num_classes'] = 14
    # config['test_start_index'] = 2
    influences = ptif.calc_img_wise(config, model, train_dataloader, test_dataloader)

    print("DONE!")