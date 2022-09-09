import sys
sys.path.append("../../")

import pickle
import json
import glob
import os
import numpy as np
from scipy import stats
import math
from tqdm import tqdm

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
        train_masks_arguments, train_ids, train_claims, train_outcomes, train_precedent, mlb = pickle.load(f)

    with open("../" + tokenized_dir + "/tokenized_test.pkl", "rb") as f:
        test_facts, test_masks, test_arguments, \
        test_masks_arguments, test_ids, test_claims, test_outcomes, test_precedent, _ = pickle.load(f)

    return train_ids, test_ids, test_precedent, train_outcomes, test_outcomes, train_claims, test_claims

def baseline_outcome(tokenized_dir, result_path, negative=False):
    train_ids, test_ids, test_precedent, train_outcomes, test_outcomes, train_claims, test_claims = initialise_data(tokenized_dir)

    negative_precedent = train_claims - train_outcomes
    negative_outcomes = test_claims - test_outcomes

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
    all_correlations = []

    for i in range(len(test_ids)):
        try:
            precedent_marked = []

            data[str(i)]['true'] = list(set(test_precedent[i]))
            data[str(i)]['id'] = test_ids[i]

            prec = []
            not_prec = []
            for j in range(len(l_dic)):
                # print(list(l_dic[j]), data[str(i)]['label'], list(l_dic[j]) == data[str(i)]['label'])
                if negative:
                    flag = list(negative_precedent[j]) == list(negative_outcomes[i])
                else:
                    flag = list(l_dic[j]) == data[str(i)]['label']

                if flag:
                    prec.append(data[str(i)]['influence'][j])
                    precedent_marked.append(1)
                else:
                    not_prec.append(data[str(i)]['influence'][j])
                    precedent_marked.append(0)

            if 1 in precedent_marked:
                # corr = np.correlate(np.array(data[str(i)]['influence'])*-1, precedent_marked)[0]
                corr = stats.spearmanr(np.array(data[str(i)]['influence']) * -1, precedent_marked)[0]
                all_correlations.append(corr)
            # print(np.correlate(precedent_marked, data[str(i)]['influence'])[0])

            # print(i, np.average(prec) > np.average(not_prec), np.average(prec), np.average(not_prec))
            # data[str(i)]['outcome_baseline'] = [np.average(prec), np.average(not_prec)]
            if len(prec) > 0 and len(not_prec) > 0:
                data[str(i)]['outcome_baseline'] = [np.average(prec), np.average(not_prec)]
                if np.average(prec) < np.average(not_prec):
                    all_pos.append(1)
                else:
                    all_neg.append(1)

        except KeyError as e:
            pass

    # print('Outcome Baseline Accuracy:', len(all_pos)/(len(all_pos)+len(all_neg)), f'{len(all_pos)}/{len(all_pos)+len(all_neg)}')
    if negative:
        print('Neg. Outcome Baseline Correlation:', np.average(all_correlations))
    else:
        print('Pos. Outcome Baseline Correlation:', np.average(all_correlations))
    print('')

def baseline_art(tokenized_dir, result_path):
    train_ids, test_ids, test_precedent, train_outcomes, _, _, _ = initialise_data(tokenized_dir)

    test_precedent = numerize_precedent(train_ids, test_precedent)
    l_dic = label_dic(train_outcomes)

    # result_path = './outdir/*'
    # result_path = find_newest(result_path)
    # result_path = './outdir/influence_results_tmp_0_False_last-i_294.json'
    # print(result_path)

    with open(result_path, 'r') as jsonFile:
        data = json.load(jsonFile)

    all_correlations = {k: [] for k in range(14)}
    all_pos = {k: [] for k in range(14)}
    all_neg = {k: [] for k in range(14)}

    for test_case in range(len(test_ids)):
        try:
            data[str(test_case)]['true'] = list(set(test_precedent[test_case]))
            data[str(test_case)]['id'] = test_ids[test_case]

            prec = {}
            not_prec = {}
            for art in range(14):
                precedent_marked = []
                prec[art] = []
                not_prec[art] = []
                for influence_i in range(len(data[str(test_case)]['influence'])):
                    if data[str(test_case)]['label'][art] == 1:
                        if list(l_dic[influence_i])[art] == 1:
                            prec[art].append(data[str(test_case)]['influence'][influence_i])
                            precedent_marked.append(1)
                        else:
                            not_prec[art].append(data[str(test_case)]['influence'][influence_i])
                            precedent_marked.append(0)

                if len(precedent_marked) > 0:
                    # corr = np.correlate(np.array(data[str(test_case)]['influence']) * -1, precedent_marked)[0]
                    corr = stats.spearmanr(np.array(data[str(test_case)]['influence']) * -1, precedent_marked)[0]
                    all_correlations[art].append(corr)

                # print(art, i, np.average(prec[art]) > np.average(not_prec[art]), np.average(prec[art]), np.average(not_prec[art]))
                # data[str(i)]['avg_baseline'] = [np.average(prec), np.average(not_prec)]
                if len(prec[art]) > 0 and len(not_prec[art]) > 0:
                    # print(art, len(prec[art]), len(not_prec[art]))
                    if np.average(prec[art]) < np.average(not_prec[art]):
                        all_pos[art].append(1)
                    else:
                        all_neg[art].append(1)

        except KeyError as e:
            pass

    results = []
    print("Article Baseline Correlation:")
    for art in range(14):
        if len(all_pos[art]) == 0 and len(all_neg[art]) == 0:
            # print(f'Per Article {art} Baseline Accuracy:', 0.0, f'{len(all_pos[art])}/{len(all_neg[art])}')
            print(f'{art} Correlation:', 'NaN')
        else:
            # print(f'Per Article {art} Baseline Accuracy:', len(all_pos[art])/(len(all_pos[art])+len(all_neg[art])), f'{len(all_pos[art])}/{len(all_pos[art])+len(all_neg[art])}')
            print(f'{art} Correlation:', np.average(all_correlations[art]))
            results.append(np.average(all_correlations[art]))

    return results

def baseline_applied(tokenized_dir, result_path):
    train_ids, test_ids, test_precedent, train_outcomes, _, _, _ = initialise_data(tokenized_dir)

    test_precedent = numerize_precedent(train_ids, test_precedent)
    l_dic = label_dic(train_outcomes)

    # result_path = './outdir/*'
    # result_path = find_newest(result_path)
    # result_path = './outdir/influence_results_tmp_0_False_last-i_294.json'

    print(result_path)

    with open(result_path, 'r') as jsonFile:
        data = json.load(jsonFile)

    applied_correlations = []
    distinguished_correlations = []
    all_pos = []
    all_neg = []
    for i in range(len(test_ids)):
        try:
            applied_marked = []
            distinguished_marked = []

            data[str(i)]['true'] = list(set(test_precedent[i]))
            data[str(i)]['id'] = test_ids[i]

            prec = []
            not_prec = []
            # for t in data[str(i)]['true']:
            #     prec.append(data[str(i)]['influence'][t])

            for j in range(len(data[str(i)]['influence'])):
                if j not in data[str(i)]['true']:
                    not_prec.append(data[str(i)]['influence'][j])
                    applied_marked.append(0)
                    distinguished_marked.append(0)
                else:
                    prec.append(data[str(i)]['influence'][j])
                    if list(l_dic[j]) == data[str(i)]['label']:
                        applied_marked.append(1)
                        distinguished_marked.append(0)
                    else:
                        distinguished_marked.append(1)
                        applied_marked.append(0)


            # corr = np.correlate(np.array(data[str(i)]['influence'])*-1, precedent_marked)[0]
            corr_app = stats.spearmanr(np.array(data[str(i)]['influence'])*-1, applied_marked)[0]
            if not math.isnan(corr_app):
                # print('app', corr_app)
                applied_correlations.append(corr_app)

            corr_neg = stats.spearmanr(np.array(data[str(i)]['influence']) * -1, distinguished_marked)[0]
            if not math.isnan(corr_neg):
                # print('dis', corr_neg)
                distinguished_correlations.append(corr_neg)

            if np.average(prec) < np.average(not_prec):
                all_pos.append(1)
                case_baseline = 1
            else:
                all_neg.append(1)
                case_baseline = -1

            # print(i, np.average(prec) > np.average(not_prec), np.average(prec), np.average(not_prec))
            data[str(i)]['avg_baseline'] = [case_baseline, np.average(prec), np.average(not_prec)]

        except KeyError as e:
            pass

    # print('Avg Baseline Accuracy:', len(all_pos)/(len(all_pos)+len(all_neg)), f'{len(all_pos)}/{len(all_pos)+len(all_neg)}')
    print('Applied Precedent Baseline Correlation:', np.average(applied_correlations))
    print('Distinguished Precedent Baseline Correlation:', np.average(distinguished_correlations))
    print('')
    # with open(result_path+'_test.json', 'w') as jsonFile:
    #     json.dump(data, jsonFile, indent=4)

def baseline_avg(tokenized_dir, result_path):
    train_ids, test_ids, test_precedent, train_outcomes, _, _, _ = initialise_data(tokenized_dir)

    test_precedent = numerize_precedent(train_ids, test_precedent)
    l_dic = label_dic(train_outcomes)

    # result_path = './outdir/*'
    # result_path = find_newest(result_path)
    # result_path = './outdir/influence_results_tmp_0_False_last-i_294.json'

    print(result_path)

    with open(result_path, 'r') as jsonFile:
        data = json.load(jsonFile)

    all_correlations = []
    all_pos = []
    all_neg = []
    for i in range(len(test_ids)):
        try:
            precedent_marked = []

            data[str(i)]['true'] = list(set(test_precedent[i]))
            data[str(i)]['id'] = test_ids[i]

            prec = []
            not_prec = []
            # for t in data[str(i)]['true']:
            #     prec.append(data[str(i)]['influence'][t])

            for j in range(len(data[str(i)]['influence'])):
                if j not in data[str(i)]['true']:
                    not_prec.append(data[str(i)]['influence'][j])
                    precedent_marked.append(0)
                else:
                    prec.append(data[str(i)]['influence'][j])
                    precedent_marked.append(1)


            # corr = np.correlate(np.array(data[str(i)]['influence'])*-1, precedent_marked)[0]
            corr = stats.spearmanr(np.array(data[str(i)]['influence'])*-1, precedent_marked)[0]
            all_correlations.append(corr)

            if np.average(prec) < np.average(not_prec):
                all_pos.append(1)
                case_baseline = 1
            else:
                all_neg.append(1)
                case_baseline = -1

            # print(i, np.average(prec) > np.average(not_prec), np.average(prec), np.average(not_prec))
            data[str(i)]['avg_baseline'] = [case_baseline, np.average(prec), np.average(not_prec)]

        except KeyError as e:
            pass

    # print('Avg Baseline Accuracy:', len(all_pos)/(len(all_pos)+len(all_neg)), f'{len(all_pos)}/{len(all_pos)+len(all_neg)}')
    print('Precedent Baseline Correlation:', np.average(all_correlations))
    print('')
    # with open(result_path+'_test.json', 'w') as jsonFile:
    #     json.dump(data, jsonFile, indent=4)


def baseline_linear(tokenized_dir, result_path):

    train_ids, test_ids, test_precedent, train_outcomes, test_outcomes, train_claims, test_claims = initialise_data(tokenized_dir)

    l_dic = label_dic(train_outcomes)


    with open(result_path, 'r') as jsonFile:
        data = json.load(jsonFile)

        all_influences = []
        all_labels = []

        for i in tqdm(range(len(data))):
            influences = []
            labels = []
            try:

                for j in range(len(l_dic)):
                    influences.append(data[str(i)]['influence'][j])

                    # print(list(l_dic[j]), data[str(i)]['label'], list(l_dic[j]) == data[str(i)]['label'])
                    flag = list(l_dic[j]) == data[str(i)]['label']

                    if flag:
                        labels.append(1)
                    else:
                        labels.append(0)

            except KeyError as e:
                pass

            all_influences.append(influences)
            all_labels.append(labels)


    return all_influences, all_labels


def influence_merge(dir_path):
    all_influences = []

    for i in range(250, 1250, 250):
        with open(dir_path + str(i) + ".json", 'r') as jsonFile:
            all_influences.append(json.load(jsonFile))

    with open(dir_path +'all.json', 'w') as outFile:
        new_dic = {}
        cnt = 0
        for split in all_influences:
            for case in split:
                new_dic[str(cnt)] = split[case]
                cnt += 1

        json.dump(new_dic, outFile, indent=2)

if __name__ == '__main__':

    tokenized_dir = "../datasets/" + 'precedent' + "/" + 'legal_bert'

    # result_path = './outdir/legal_bert/facts/f1f984acd35b4283947585cac74ed6bd_0_250_0_False_last-i_203.json'
    # # result_path = './outdir/bert/facts/influence_results_tmp_0_False_last-i_294.json'
    # # result_path = './outdir/bert/facts/927927e50ca941ceb7a0b09b51fe54fb_0_250_0_False_last-i_249.json'
    # # result_path = './outdir/bert/both/987cd7bdc92b42afab32772c509aa246_0_10000_0_False_last-i_291.json'

    # baseline_linear(tokenized_dir, result_path)
    # influence_merge('./outdir/legal_bert/facts/')

    result_path = './outdir/legal_bert/facts/all.json'
    baseline_applied(tokenized_dir, result_path)
    baseline_avg(tokenized_dir, result_path)
    baseline_outcome(tokenized_dir, result_path)
    baseline_outcome(tokenized_dir, result_path, True)
    baseline_art(tokenized_dir, result_path)