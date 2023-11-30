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
import csv
import ast


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


def check_cited(cited, precedent_dic, i, j):
    if cited:
        if j in precedent_dic[i]:
            cited_constrain = True
        else:
            cited_constrain = False
    else:
        cited_constrain = True

    return cited_constrain


def taxonomy(cited_constrain, negative_precedent, negative_outcomes, test_outcomes, train_outcomes, test_claims,
             train_claims, positive_applied, negative_applied, positive_distinguished, negative_distinguished, i, j,
             art_range):

    n_p = negative_precedent[j].tolist()[art_range[0]: art_range[1]]
    n_o = negative_outcomes[i].tolist()[art_range[0]: art_range[1]]
    t_o = train_outcomes[j].tolist()[art_range[0]: art_range[1]]
    test_o = test_outcomes[i].tolist()[art_range[0]: art_range[1]]
    t_c = train_claims[j].tolist()[art_range[0]: art_range[1]]
    test_c = test_claims[i].tolist()[art_range[0]: art_range[1]]

    if cited_constrain:
        # negative applied
        if 1 in n_p and n_p == n_o:
            negative_applied.append(1)
        else:
            negative_applied.append(0)
        # positive applied
        if 1 in t_o and t_o == test_o:
            positive_applied.append(1)
        else:
            positive_applied.append(0)

        # distingishing, we only consider cases where precedent shares claims
        if 1 in t_c and t_c == test_c:
            # print("CB:", train_claims[j])
            # print("PO:", test_outcomes[i])
            # print("NO:", negative_outcomes[i])
            # print("NP:", negative_precedent[j])
            # print("PP:", train_outcomes[j])
            # positive
            if 1 in n_p and n_p == test_o:
                negative_distinguished.append(1)
                # print("+-")
            else:
                negative_distinguished.append(0)
            # negative
            if 1 in t_o and t_o == n_o:
                # print("-+")
                positive_distinguished.append(1)
            else:
                positive_distinguished.append(0)
            # print("\n")
        else:
            positive_distinguished.append(0)
            negative_distinguished.append(0)
    else:
        positive_applied.append(0)
        negative_applied.append(0)
        positive_distinguished.append(0)
        negative_distinguished.append(0)

    return positive_applied, negative_applied, positive_distinguished, negative_distinguished

def cnt_examples(precedents):
    precedents = np.array(precedents)
    return len(precedents[precedents==1])

def create_baseline(split, size):
    baseline = np.random.random(size)
    baseline[baseline > split] = 1
    baseline[baseline < split] = 0
    return baseline

def baseline_outcome(tokenized_dir, result_path, cited=False, art_range=None, all_predictions=False, all_model_predictions=False):

    if art_range is None: art_range = [0, 100]

    train_ids, test_ids, test_precedent, train_outcomes, test_outcomes, train_claims, test_claims = initialise_data(
        tokenized_dir)

    correct_results = []

    with open(result_path.split('all.json')[0] + 'outputs.csv') as f:
        results = csv.reader(f)
        for row in results:
            correct_results.append(row[1] == row[2])

    if not all_predictions and all_model_predictions:
        return AssertionError("Invalid combination: You have to use 'all_preidctions' for 'all_model_predictions'")

    if all_predictions:
        print('Using All Predictions')
    else:
        print('Using only predictions that the model got right.')

    if all_model_predictions:
        print('Using model predictions as the source for precedent taxonomy.')
        with open(result_path.split('all.json')[0] + 'outputs.csv') as f:
            results = list(csv.reader(f))

        x = np.array([ast.literal_eval(row[1]) for row in results])

        if "joint" in result_path:
            test_outcomes = x.copy()
            test_outcomes[test_outcomes==2] = 0
            negative_outcomes = x.copy()
            negative_outcomes[negative_outcomes == 1] = 0
            negative_outcomes[negative_outcomes == 2] = 1
        else:
            test_outcomes = x
            negative_outcomes = test_claims - test_outcomes
            negative_outcomes[negative_outcomes == -1] = 0

    else:
        print('Using judges predictions as the source for precedent taxonomy.')
        negative_outcomes = test_claims - test_outcomes

    negative_precedent = train_claims - train_outcomes

    with open(result_path, 'r') as jsonFile:
        results = json.load(jsonFile)

    precedent_dic = numerize_precedent(train_ids, test_precedent)

    all_positive_applied = []
    all_negative_applied = []
    all_positive_distinguished = []
    all_negative_distinguished = []
    all_influences = []
    all_any = []

    all_inf_positive_applied = []
    all_inf_negative_applied = []
    all_inf_positive_distinguished = []
    all_inf_negative_distinguished = []

    for i in range(len(test_ids)):

        if correct_results[i] or all_predictions:

            positive_applied = []
            negative_applied = []
            positive_distinguished = []
            negative_distinguished = []
            any_type = []

            for j in range(len(train_outcomes)):
                cited_constrain = check_cited(cited, precedent_dic, i, j)
                positive_applied, negative_applied, positive_distinguished, negative_distinguished = taxonomy(cited_constrain, negative_precedent, negative_outcomes, test_outcomes, train_outcomes, test_claims, train_claims, positive_applied, negative_applied, positive_distinguished, negative_distinguished, i, j, art_range)

            all_positive_applied += positive_applied
            all_negative_applied += negative_applied
            all_positive_distinguished += positive_distinguished
            all_negative_distinguished += negative_distinguished

            # if 1 in positive_applied:
            #     all_positive_applied += positive_applied
            #     all_inf_positive_applied += list(np.array(results[str(i)]['influence'])*-1)
            # if 1 in negative_applied:
            #     all_negative_applied += negative_applied
            #     all_inf_negative_applied += list(np.array(results[str(i)]['influence'])*-1)
            # if 1 in positive_distinguished:
            #     all_positive_distinguished += positive_distinguished
            #     all_inf_positive_distinguished += list(np.array(results[str(i)]['influence'])*-1)
            # if 1 in negative_distinguished:
            #     all_negative_distinguished += negative_distinguished
            #     all_inf_negative_distinguished += list(np.array(results[str(i)]['influence'])*-1)

            any_type = np.array(positive_applied) + np.array(negative_applied) + np.array(positive_distinguished) + np.array(negative_distinguished)
            any_type[any_type > 1] = 1
            all_any += any_type.tolist()

            # if 1 not in any_type:
            #     print("Error")

            # print(1 in positive_applied, 1 in negative_applied, 1 in positive_distinguished, 1 in negative_distinguished)

            all_influences += list(np.array(results[str(i)]['influence'])*-1)

            # if 1 in positive_applied:
            #     # corr = np.correlate(np.array(data[str(i)]['influence'])*-1, precedent_marked)[0]
            #     corr = stats.spearmanr(np.array(results[str(i)]['influence']) * -1, positive_applied)[0]
            #     all_positive_applied.append(corr)
            #     # all_positive_applied += positive_applied
            #
            # if 1 in negative_applied:
            #     corr = stats.spearmanr(np.array(results[str(i)]['influence']) * -1, negative_applied)[0]
            #     all_negative_applied.append(corr)
            #     # all_negative_applied += negative_applied
            #
            # if 1 in positive_distinguished:
            #     corr = stats.spearmanr(np.array(results[str(i)]['influence']) * -1, positive_distinguished)[0]
            #     all_positive_distinguished.append(corr)
            #     # all_positive_distinguished += positive_distinguished
            #
            # if 1 in negative_distinguished:
            #     corr = stats.spearmanr(np.array(results[str(i)]['influence']) * -1, negative_distinguished)[0]
            #     all_negative_distinguished.append(corr)
            #     # all_negative_distinguished += negative_distinguished

    # corr_positive_applied = np.average(all_positive_applied)
    # corr_negative_applied = np.average(all_negative_applied)
    # corr_positive_distinguished = np.average(all_positive_distinguished)
    # corr_negative_distinguished = np.average(all_negative_distinguished)

    size = len(all_any)
    baseline = create_baseline(0.9, size)
    baseline_10 = stats.spearmanr(all_influences, baseline)[0]
    baseline = create_baseline(0.5, size)
    baseline_50 = stats.spearmanr(all_influences, baseline)[0]

    corr_positive_applied = stats.spearmanr(all_influences, all_positive_applied)[0]
    corr_negative_applied = stats.spearmanr(all_influences, all_negative_applied)[0]
    corr_positive_distinguished = stats.spearmanr(all_influences, all_positive_distinguished)[0]
    corr_negative_distinguished = stats.spearmanr(all_influences, all_negative_distinguished)[0]
    corr_any = stats.spearmanr(all_influences, all_any)[0]

    # corr_positive_applied = stats.spearmanr(all_inf_positive_applied, all_positive_applied)[0]
    # corr_negative_applied = stats.spearmanr(all_inf_negative_applied, all_negative_applied)[0]
    # corr_positive_distinguished = stats.spearmanr(all_inf_positive_distinguished, all_positive_distinguished)[0]
    # corr_negative_distinguished = stats.spearmanr(all_inf_negative_distinguished, all_negative_distinguished)[0]
    # corr_any = stats.spearmanr(all_influences, all_any)[0]

    # print('Positive Applied Correlation:', np.average(all_positive_applied))
    # print('Negative Applied Correlation:', np.average(all_negative_applied))
    # print('Positive Distinguished Correlation:', np.average(all_positive_distinguished))
    # print('Negative Distinguished Correlation:', np.average(all_negative_distinguished))

    print('Baseline 50 Correlation:', baseline_50)
    print('Baseline 10 Correlation:', baseline_10)
    print('Positive Applied Correlation:', corr_positive_applied, cnt_examples(all_positive_applied))
    print('Negative Applied Correlation:', corr_negative_applied, cnt_examples(all_negative_applied))
    print('Positive Distinguished Correlation:', corr_positive_distinguished, cnt_examples(all_positive_distinguished))
    print('Negative Distinguished Correlation:', corr_negative_distinguished, cnt_examples(all_negative_distinguished))
    print('Any Correlation:', corr_any, cnt_examples(all_any))

    # model_influences = all_influences + all_influences + all_influences + all_influences
    # model_precedents = all_positive_applied + all_negative_applied + all_positive_distinguished + all_negative_distinguished
    # model_precedents = np.array(all_positive_applied) + np.array(all_negative_applied) + np.array(all_positive_distinguished) + np.array(all_negative_distinguished)
    # model_precedents[model_precedents > 1] = 1
    # print('Correlation All:', stats.spearmanr(all_influences, model_precedents)[0])


    return [corr_positive_applied, corr_negative_applied, corr_positive_distinguished, corr_negative_distinguished, corr_any], all_influences, all_any
    # return None, None, None


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
    train_ids, test_ids, test_precedent, train_outcomes, test_outcomes, train_claims, test_claims = initialise_data(
        tokenized_dir)
    # train_ids, test_ids, test_precedent, train_outcomes, _, _, _ = initialise_data(tokenized_dir)

    negative_precedent = train_claims - train_outcomes
    negative_outcomes = test_claims - test_outcomes

    # test_precedent = numerize_precedent(train_ids, test_precedent)
    # l_dic = label_dic(train_outcomes)

    test_precedent = numerize_precedent(train_ids, test_precedent)
    l_dic = label_dic(train_outcomes)
    claim_dic = label_dic(train_claims)
    neg_dic = label_dic(negative_precedent)

    # result_path = './outdir/*'
    # result_path = find_newest(result_path)
    # result_path = './outdir/influence_results_tmp_0_False_last-i_294.json'

    print(result_path)

    with open(result_path, 'r') as jsonFile:
        data = json.load(jsonFile)

    applied_correlations = []
    distinguished_correlations = []
    negative_correlations = []
    all_pos = []
    all_neg = []
    for i in range(len(test_ids)):
        try:
            applied_marked = []
            distinguished_marked = []
            negative_marked = []

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
                    negative_marked.append(0)
                else:
                    prec.append(data[str(i)]['influence'][j])
                    if list(l_dic[j]) == data[str(i)]['label']:
                        applied_marked.append(1)
                        distinguished_marked.append(0)
                    else:
                        distinguished_marked.append(1)
                        applied_marked.append(0)
                    if list(neg_dic[j]) == data[str(i)]['label']:
                        negative_marked.append(1)
                    else:
                        negative_marked.append(0)

            # corr = np.correlate(np.array(data[str(i)]['influence'])*-1, precedent_marked)[0]
            corr_app = stats.spearmanr(np.array(data[str(i)]['influence']) * -1, applied_marked)[0]
            if not math.isnan(corr_app):
                # print('app', corr_app)
                applied_correlations.append(corr_app)

            corr_neg = stats.spearmanr(np.array(data[str(i)]['influence']) * -1, distinguished_marked)[0]
            if not math.isnan(corr_neg):
                # print('dis', corr_neg)
                distinguished_correlations.append(corr_neg)

            corr_negative = stats.spearmanr(np.array(data[str(i)]['influence']) * -1, negative_marked)[0]
            if not math.isnan(corr_negative):
                # print('dis', corr_neg)
                negative_correlations.append(corr_negative)

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
    print('Negative Precedent Baseline Correlation:', np.average(negative_correlations))
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
            corr = stats.spearmanr(np.array(data[str(i)]['influence']) * -1, precedent_marked)[0]
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
    train_ids, test_ids, test_precedent, train_outcomes, test_outcomes, train_claims, test_claims = initialise_data(
        tokenized_dir)

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

                    if 'joint' in result_path:
                        converted = np.array(data[str(i)]['label'])
                        converted[converted == 2] = 0
                        flag = list(l_dic[j]) == converted.tolist()
                    else:
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

    for i in range(50, 1000, 50):
        with open(dir_path + str(i) + ".json", 'r') as jsonFile:
            all_influences.append(json.load(jsonFile))

    with open(dir_path + 'all.json', 'w') as outFile:
        new_dic = {}
        cnt = 0
        for split in all_influences:
            for case in split:
                new_dic[str(cnt)] = split[case]
                cnt += 1

        json.dump(new_dic, outFile, indent=2)

def average_scores(tokenized_dir, result_path):
    train_ids, test_ids, test_precedent, train_outcomes, test_outcomes, train_claims, test_claims = initialise_data(
        tokenized_dir)
    test_precedent = numerize_precedent(train_ids, test_precedent)

    with open(result_path, 'r') as jsonFile:
        results = json.load(jsonFile)

    precedent_influences = []
    other_influences = []
    for i in range(len(test_ids)):
        tp = test_precedent[i]
        for j in range(len(train_outcomes)):
            influence = results[str(i)]['influence'][j]
            if j in tp:
                precedent_influences.append(influence*-1)
            else:
                other_influences.append(influence*-1)

    print("cited:", np.average(precedent_influences))
    print("not cited:", np.average(other_influences))

def correlation_run(tokenized_dir, result_path, all_predictions=False, all_model_predictions=False):
    all_correlations, all_influences, all_precedents = {}, [], []
    # print("average")
    # average_scores(tokenized_dir, result_path)
    print("wide")
    correlations, all_influences, wide_precedents = baseline_outcome(tokenized_dir, result_path, False, None, all_predictions, all_model_predictions)
    all_correlations["wide"] = {'pos_applied': correlations[0], 'neg_applied': correlations[1],
                                  "pos_distinguished": correlations[2], "neg_distinguished": correlations[3], "all": correlations[4]}
    print("narrow")
    correlations, all_influences, narrow_precedents = baseline_outcome(tokenized_dir, result_path, True, None, all_predictions, all_model_predictions)
    all_correlations["narrow"] = {'pos_applied':correlations[0], 'neg_applied':correlations[1],"pos_distinguished":correlations[2], "neg_distinguished":correlations[3], "all": correlations[4]}

    all_precedents = np.array(wide_precedents) + np.array(narrow_precedents)
    all_precedents[all_precedents > 1] = 1
    all_precedents = all_precedents.tolist()
    print('Overall Correlation SPEARMAN:', stats.spearmanr(all_influences, all_precedents)[0])
    all_correlations["all"] = stats.spearmanr(all_influences, all_precedents)[0]
    print('Overall Correlation PEARSON:', stats.pearsonr(all_influences, all_precedents)[0])

    return all_correlations

if __name__ == '__main__':
    # tokenized_dir = "../datasets/" + 'precedent' + "/" + 'legal_bert'

    # result_path = './outdir/legal_bert/facts/f1f984acd35b4283947585cac74ed6bd_0_250_0_False_last-i_203.json'
    # # result_path = './outdir/bert/facts/influence_results_tmp_0_False_last-i_294.json'
    # # result_path = './outdir/bert/facts/927927e50ca941ceb7a0b09b51fe54fb_0_250_0_False_last-i_249.json'
    # # result_path = './outdir/bert/both/987cd7bdc92b42afab32772c509aa246_0_10000_0_False_last-i_291.json'

    tokenized_dir = "../datasets/" + 'precedent' + "/" + 'legal_bert'
    result_path = './outdir/joint/legal_bert/facts/all.json'
    baseline_linear(tokenized_dir, result_path)
    # --------------

    # influence_merge('./outdir/joint/legal_bert/facts/')
    #
    # result_path = './outdir/legal_bert/facts/all.json'
    # baseline_applied(tokenized_dir, result_path)
    # baseline_avg(tokenized_dir, result_path)

    tokenized_dir = "../datasets/" + 'precedent' + "/" + 'legal_bert'
    result_path = './outdir/joint/legal_bert/facts/all.json'
    print("\n joint_legal_bert")
    correlation_run(tokenized_dir, result_path, all_predictions=True, all_model_predictions=False)

    tokenized_dir = "../datasets/" + 'precedent' + "/" + 'legal_bert'
    result_path = './outdir/legal_bert/facts/all.json'
    print("\n legal_bert")
    correlation_run(tokenized_dir, result_path, all_predictions=True, all_model_predictions=False)

    tokenized_dir = "../datasets/" + 'precedent' + "/" + 'bert'
    result_path = './outdir/joint/bert/facts/all.json'
    print("\n joint_bert")
    correlation_run(tokenized_dir, all_predictions=True, all_model_predictions=False)

    tokenized_dir = "../datasets/" + 'precedent' + "/" + 'bert'
    result_path = './outdir/bert/facts/all.json'
    print("\n bert")
    correlation_run(tokenized_dir, all_predictions=True, all_model_predictions=False)

