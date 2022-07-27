import pytorch_influence_functions as ptif
import torch
import sys
from src.train.data_loader import DataPrep
import pickle
import json
import glob
import os
import argparse

def find_newest(path):
    list_of_files = glob.glob(path)
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def numerize_precedent(train_ids, test_precedent):
    new_precedent = []
    for prec in test_precedent:
        new_prec = []
        for p in prec:
            for i in range(len(train_ids)):
                if p in train_ids[i]:
                    new_prec.append(i)
        new_precedent.append(new_prec)

    return new_precedent

def label_results(tokenized_dir, test):
    with open("../" + tokenized_dir + "/tokenized_train.pkl", "rb") as f:
        train_facts, train_masks, train_arguments, \
        train_masks_arguments, train_ids, train_claims, train_outcomes, train_precedent, _ = pickle.load(f)

    with open("../" + tokenized_dir + "/tokenized_test.pkl", "rb") as f:
        test_facts, test_masks, test_arguments, \
        test_masks_arguments, test_ids, test_claims, test_outcomes, test_precedent, _ = pickle.load(f)

    if test:
        train_t = 6
        test_t = 2
    else:
        train_t = 100000
        test_t = 100000

    test_precedent = numerize_precedent(train_ids, test_precedent)

    train_ids = train_ids[:train_t]
    test_ids = test_ids[:test_t]
    test_precedent = test_precedent[:test_t]

    result_path = './outdir/*'
    result_path = find_newest(result_path)

    with open(result_path, 'r') as jsonFile:
        data = json.load(jsonFile)

    for i in range(len(test_ids)):
        data[str(i)]['true'] = test_precedent[i]
        data[str(i)]['id'] = test_ids[i]

    with open(result_path, 'w') as jsonFile:
        json.dump(data, jsonFile, indent=4)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=512, required=False)
    parser.add_argument("--batch_size", type=int, default=16, required=False)
    parser.add_argument("--test", dest='test', action='store_true')
    args = parser.parse_args()

    sys.path.insert(0, '../train')
    # find the last trained model
    model_path = '../train/trained_models/precedent/bert/facts/*'
    model_path = find_newest(model_path)
    model = torch.load(model_path + '/model.pt')

    tokenized_dir = "../datasets/" + 'precedent' + "/" + 'bert'
    # tokenizer_dir, test, log, max_len, batch_size
    loader = DataPrep(tokenized_dir, args.test, None, args.max_len, args.batch_size)
    train_dataloader, val_dataloader, test_dataloader = loader.load()

    ptif.init_logging()
    config = ptif.get_default_config()
    config['gpu'] = -1
    config['test_sample_num'] = False
    influences = ptif.calc_img_wise(config, model, train_dataloader, test_dataloader)

    label_results(tokenized_dir, args.test)