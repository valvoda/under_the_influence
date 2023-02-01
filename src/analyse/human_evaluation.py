import json
import random

from baselines import initialise_data
from src.preprocess.preprocess_data import get_arguments, fix_claims
import re
import os
from tqdm import tqdm
from pathlib import Path
import csv

def get_data(pretokenized_dir):
    dataset = {}
    paths = ['train', 'dev', 'test']
    out_path = ['train_augmented', 'dev_augmented', 'test_augmented']

    for case_path, out in zip(paths, out_path):

        all = {'facts': [],
               'arguments': [],
               'claims': [],
               'outcomes': [],
               'ids': [],
               'precedent': [],
               'dates': []}

        for item in tqdm(os.listdir("../../datasets/precedent/"+case_path)):
            if item.endswith('.json'):
                with open(os.path.join("../../datasets/precedent/"+case_path, item), "r") as json_file:
                    data = json.load(json_file)
                    try:
                        alleged_arguments = data["text"].split("THE LAW")[1].split("FOR THESE REASONS, THE COURT UNANIMOUSLY")[0].lower()
                        convention_claims = list(set(re.findall("article\s(\d{1,2})\s.{0,15}convention", alleged_arguments)))
                    except:
                        convention_claims = []

                    try:
                        alleged_arguments = data["text"].split("THE LAW")[1].split("FOR THESE REASONS, THE COURT UNANIMOUSLY")[0].lower()
                        protocol_claims = list(set(re.findall("article\s(\d{1,2})\s.{0,15}protocol.{0,15}(\d)", alleged_arguments)))
                        protocol_claims = ['P' + p[1] + "-" + p[0] for p in protocol_claims]
                    except:
                        protocol_claims = []

                    argument = get_arguments(data)
                    claims = list(set(convention_claims + protocol_claims))
                    data['claim'] = claims
                    data['arguments'] = argument

                if len(claims) > 0 and len(data['citations']) > 0:
                    all['facts'].append(data["facts"])
                    all['claims'].append(claims)
                    all['arguments'].append(argument)
                    all['outcomes'].append(data["violated_articles"])
                    all['ids'].append(str(data["case_no"]))
                    all['precedent'].append(data["citations"])
                    all['dates'].append(data["judgment_date"])

        all = fix_claims(all)
        print(pretokenized_dir, len(all['ids']))
        dataset[case_path] = all

    return dataset

def generate_he():
    with open('./outdir/bert/facts/250.json', 'r') as jsonFile:
        influence_data = json.load(jsonFile)

    tokenized_dir = "../datasets/" + 'precedent' + "/" + 'bert'
    train_ids, test_ids, test_precedent, train_outcomes, test_outcomes, _, _ = initialise_data(tokenized_dir)

    dataset = get_data("../datasets/precedent/bert")

    fact_train_dict = dict(zip(dataset['train']['ids'], dataset['train']['facts']))
    arg_train_dict = dict(zip(dataset['train']['ids'], dataset['train']['arguments']))
    outcome_train_dict = dict(zip(dataset['train']['ids'], dataset['train']['claims']))
    fact_test_dict = dict(zip(dataset['test']['ids'], dataset['test']['facts']))
    outcome_test_dict = dict(zip(dataset['test']['ids'], dataset['test']['claims']))

    inf_he_dict = {}
    names_he_dict = {}

    for i in range(100):

        Path('../../datasets/human_evaluation/'+str(i)).mkdir(parents=True, exist_ok=True)

        case_facts = fact_test_dict[test_ids[i]]

        with open('../../datasets/human_evaluation/'+str(i)+'/case_facts.txt', 'w') as f:
            f.write(test_ids[i] + "\n")
            for fact in case_facts:
                f.write(fact + "\n")

        infs, ids = zip(*sorted(zip(influence_data[str(i)]['influence'], train_ids)))

        worst = ids[-1]
        best = ids[0]
        mid = ids[len(ids)//2]

        worst_inf = infs[-1]
        best_inf = infs[0]
        mid_inf = infs[len(infs)//2]

        inf_he_dict[i] = [worst_inf, best_inf, mid_inf]
        names_he_dict[i] = [worst.split('/')[0], best.split('/')[0], mid.split('/')[0]]

        all = [best, worst, mid]
        name = ['best', 'worst', 'mid']

        c = list(zip(all, name))
        random.shuffle(c)
        all, name = zip(*c)

        print(outcome_test_dict[test_ids[i]])
        # for fact in case_facts:
            # print(fact)
        for a, n in zip(all, name):
            print(n)
            print(outcome_train_dict[a])
            train_facts = fact_train_dict[a]
            train_args = arg_train_dict[a]

            with open('../../datasets/human_evaluation/' + str(i) + '/' + a.split('/')[0] + '.txt', 'w') as f:
                f.write(a + "\n")
                f.write("FACTS\n")
                for fact in train_facts:
                    f.write(fact + "\n")
                f.write("\nARGUMENTS\n")
                for arg in train_args:
                    f.write(arg + "\n")

            # for fact in test_facts:
                # print(fact)
            # print(test_args)

    header = ['id', 'candidate_1', 'candidate_2', 'candidate_3']
    with open('../../datasets/human_evaluation/ranking_sheet.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for k, v in zip(names_he_dict.keys(), names_he_dict.values()):
            k = [k]
            k += v
            writer.writerow(k)
            writer.writerow(['ranking'])

    with open('../../datasets/human_evaluation/truth_sheet.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for k, v in zip(inf_he_dict.keys(), inf_he_dict.values()):
            k = [k]
            k += v
            writer.writerow(k)

    print('Done')

if __name__ == '__main__':
    generate_he()