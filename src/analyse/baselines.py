import sys
sys.path.append("../../")

import pytorch_influence_functions as ptif
import torch
from src.preprocess.data_loader import DataPrep
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

def label_dic(train_outcomes):
    l_dic = {}
    cnt = 0
    for i in train_outcomes:
        l_dic[cnt] = i
        cnt += 1

    return l_dic

def initialise_data(tokenized_dir):
    with open("../" + tokenized_dir + "/tokenized_train.pkl", "rb") as f:
        train_facts, train_masks, train_arguments, \
        train_masks_arguments, train_ids, train_claims, train_outcomes, train_precedent, _ = pickle.load(f)

    with open("../" + tokenized_dir + "/tokenized_test.pkl", "rb") as f:
        test_facts, test_masks, test_arguments, \
        test_masks_arguments, test_ids, test_claims, test_outcomes, test_precedent, _ = pickle.load(f)

    return train_ids, test_ids, test_precedent, train_outcomes

def baseline_outcome(tokenized_dir, result_path):
    train_ids, test_ids, test_precedent, train_outcomes = initialise_data(tokenized_dir)

    test_precedent = numerize_precedent(train_ids, test_precedent)
    l_dic = label_dic(train_outcomes)

    # result_path = './outdir/*'
    # result_path = find_newest(result_path)
    # result_path = './outdir/influence_results_tmp_0_False_last-i_294.json'
    # print(result_path)

    with open(result_path, 'r') as jsonFile:
        data = json.load(jsonFile)

    all_pos = []
    all_neg = []

    for i in range(len(test_ids)):
        try:
            data[str(i)]['true'] = list(set(test_precedent[i]))
            data[str(i)]['id'] = test_ids[i]

            prec = []
            not_prec = []
            for j in range(len(l_dic)):
                print(list(l_dic[j]), data[str(i)]['label'], list(l_dic[j]) == data[str(i)]['label'])
                if list(l_dic[j]) == data[str(i)]['label']:
                    prec.append(data[str(i)]['influence'][j])
                else:
                    not_prec.append(data[str(i)]['influence'][j])

            # print(i, np.average(prec) > np.average(not_prec), np.average(prec), np.average(not_prec))
            data[str(i)]['outcome_baseline'] = [np.average(prec), np.average(not_prec)]
            if np.average(prec) > np.average(not_prec):
                all_pos.append(1)
            else:
                all_neg.append(1)

        except:
            pass

    print('Outcome Baseline Accuracy:', len(all_pos)/(len(all_pos)+len(all_neg)), f'{len(all_pos)}/{len(all_pos)+len(all_neg)}')


def baseline_art(tokenized_dir, result_path):
    train_ids, test_ids, test_precedent, train_outcomes = initialise_data(tokenized_dir)

    test_precedent = numerize_precedent(train_ids, test_precedent)
    l_dic = label_dic(train_outcomes)

    # result_path = './outdir/*'
    # result_path = find_newest(result_path)
    # result_path = './outdir/influence_results_tmp_0_False_last-i_294.json'
    # print(result_path)

    with open(result_path, 'r') as jsonFile:
        data = json.load(jsonFile)

    all_pos = {k: [] for k in range(14)}
    all_neg = {k: [] for k in range(14)}

    for i in range(len(test_ids)):
        try:
            data[str(i)]['true'] = list(set(test_precedent[i]))
            data[str(i)]['id'] = test_ids[i]

            prec = {}
            not_prec = {}
            for art in range(14):
                prec[art] = []
                not_prec[art] = []
                for j in range(len(data[str(i)]['influence'])):
                    if data[str(i)]['label'][art] == 1:
                        if list(l_dic[j])[art] == 1:
                            prec[art].append(data[str(i)]['influence'][j])
                        else:
                            not_prec[art].append(data[str(i)]['influence'][j])

                # print(art, i, np.average(prec[art]) > np.average(not_prec[art]), np.average(prec[art]), np.average(not_prec[art]))
                # data[str(i)]['avg_baseline'] = [np.average(prec), np.average(not_prec)]
                if len(prec[art]) > 0 or len(not_prec[art]) > 0:
                    if np.average(prec[art]) > np.average(not_prec[art]):
                        all_pos[art].append(1)
                    else:
                        all_neg[art].append(1)
        except:
            pass

    for art in range(14):
        if len(all_pos[art]) == 0 and len(all_neg[art]) == 0:
            print(f'Per Article {art} Baseline Accuracy:', 0.0, f'{len(all_pos[art])}/{len(all_neg[art])}')
        else:
            print(f'Per Article {art} Baseline Accuracy:', len(all_pos[art])/(len(all_pos[art])+len(all_neg[art])), f'{len(all_pos[art])}/{len(all_pos[art])+len(all_neg[art])}')

def baseline_avg(tokenized_dir, result_path):
    train_ids, test_ids, test_precedent, train_outcomes = initialise_data(tokenized_dir)

    test_precedent = numerize_precedent(train_ids, test_precedent)
    l_dic = label_dic(train_outcomes)

    # result_path = './outdir/*'
    # result_path = find_newest(result_path)
    # result_path = './outdir/influence_results_tmp_0_False_last-i_294.json'

    print(result_path)

    with open(result_path, 'r') as jsonFile:
        data = json.load(jsonFile)

    all_pos = []
    all_neg = []
    for i in range(len(test_ids)):
        try:
            data[str(i)]['true'] = list(set(test_precedent[i]))
            data[str(i)]['id'] = test_ids[i]

            prec = []
            not_prec = []
            for t in data[str(i)]['true']:
                prec.append(data[str(i)]['influence'][t])

            for j in range(len(data[str(i)]['influence'])):
                if j not in data[str(i)]['true']:
                    not_prec.append(data[str(i)]['influence'][i])

            if np.average(prec) > np.average(not_prec):
                all_pos.append(1)
                case_baseline = 1
            else:
                all_neg.append(1)
                case_baseline = -1

            # print(i, np.average(prec) > np.average(not_prec), np.average(prec), np.average(not_prec))
            data[str(i)]['avg_baseline'] = [case_baseline, np.average(prec), np.average(not_prec)]

        except:
            pass

    print('Avg Baseline Accuracy:', len(all_pos)/(len(all_pos)+len(all_neg)), f'{len(all_pos)}/{len(all_pos)+len(all_neg)}')

    # with open(result_path+'_test.json', 'w') as jsonFile:
    #     json.dump(data, jsonFile, indent=4)

if __name__ == '__main__':

    tokenized_dir = "../datasets/" + 'precedent' + "/" + 'bert'
    result_path = './outdir/influence_results_tmp_0_False_last-i_61_2022-08-03-09-28-47.json'
    # baseline_avg(tokenized_dir, result_path)
    baseline_outcome(tokenized_dir, result_path)
    # baseline_art(tokenized_dir, result_path)