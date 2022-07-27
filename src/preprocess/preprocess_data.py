from tqdm import tqdm
import os
import json
import re
import pickle
from pathlib import Path
import numpy as np
from collections import Counter

import torch
from transformers import LongformerTokenizer, BertTokenizer, AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


def fix_claims(all):

    new = {'facts': [],
           'arguments': [],
           'claims': [],
           'outcomes': [],
           'ids': [],
           'precedent': [],
           'dates': []}

    allowed = ['10', '11', '13', '14', '2', '3', '5', '6', '7', '8', '9', 'P1-1', 'P1-3', 'P4-2']

    for claim, outcome, c_id, fact, argument, precedent, date in tqdm(zip(all['claims'], all['outcomes'], all['ids'], all['facts'], all['arguments'], all['precedent'], all['dates'])):
        n_c = []
        for c in claim:
            if c in allowed:
                n_c.append(c)

        flag = True
        if len(n_c) > 0 and len(n_c) >= len(outcome):
            for o in outcome:
                if o not in n_c:
                    flag = False

            if flag:
                n_c.sort()
                outcome.sort()
                new['claims'].append(n_c)
                new['outcomes'].append(outcome)
                new['ids'].append(c_id)
                new['facts'].append(fact)
                new['arguments'].append(argument)
                new['dates'].append(date)
                new['precedent'].append(precedent)

    return new


def fix_citations(dataset):

    flat = [c.split(";") for c in dataset['train']['ids']]
    train_case_ids = []
    for f in flat:
        for i in f:
            train_case_ids.append(i)
    train_case_ids = list(set(train_case_ids))

    for i in ['dev', 'test']:
        print(len(dataset[i]['ids']))
        dataset[i], cnt_removed = remove_citations(dataset[i], train_case_ids)
        print(i, cnt_removed)
        print(len(dataset[i]['ids']))

    return dataset


def remove_citations(all, allowed):
    new = {'facts': [],
           'arguments': [],
           'claims': [],
           'outcomes': [],
           'ids': [],
           'precedent': [],
           'dates': []}

    cnt = 0
    for claim, outcome, c_id, fact, argument, precedent, date in tqdm(
            zip(all['claims'], all['outcomes'], all['ids'], all['facts'], all['arguments'], all['precedent'],
                all['dates'])):
        n_p = []
        for p in precedent:
            if p in allowed:
                n_p.append(p)

        if len(n_p) > 0:
            n_p.sort()
            new['claims'].append(claim)
            new['outcomes'].append(outcome)
            new['ids'].append(c_id)
            new['facts'].append(fact)
            new['arguments'].append(argument)
            new['dates'].append(date)
            new['precedent'].append(n_p)
        else:
            cnt += 1

    flat = []
    for p in all['precedent']:
        for i in p:
            flat.append(i)

    cntr_dic = Counter(flat)
    print(len(cntr_dic))

    return new, cnt

def get_arguments(data):
    try:
        arguments = data["text"].split("THE LAW")[1].split("FOR THESE REASONS, THE COURT UNANIMOUSLY")[0]
        arguments = arguments.split("\n")
        arguments = [a.strip() for a in arguments]
        arguments = list(filter(None, arguments))
    except:
        return []
    return arguments

def get_data(pretokenized_dir, tokenizer, max_len):
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

        for item in tqdm(os.listdir("datasets/precedent/"+case_path)):
            if item.endswith('.json'):
                with open(os.path.join("datasets/precedent/"+case_path, item), "r") as json_file:
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

                with open(os.path.join('datasets/precedent/'+out, item), "w") as out_file:
                    json.dump(data, out_file, indent=1)

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

    dataset = fix_citations(dataset)

    process_dataset(pretokenized_dir, tokenizer, max_len, dataset)

def get_stats(data):
    data = np.array(data)
    stats = np.array([0 for i in range(len(data[0]))])
    cnt = 0
    for d in data:
        stats = stats + d
        if d.sum() > 0:
            cnt += 1

    return stats, cnt

def get_neg(claim_data, out_data):
    cdata = np.array(claim_data)
    odata = np.array(out_data)
    c_stats, c_cnt = get_stats(claim_data)
    out_stats, out_cnt = get_stats(out_data)
    stats = c_stats - out_stats
    cnt = 0
    for c, o in zip(cdata, odata):
        n = c - o
        if n.sum() > 0:
            cnt += 1

    return stats, cnt

def data_stats(claims, outcomes, type):
    c_stats, c_cnt = get_stats(claims)
    out_stats, out_cnt = get_stats(outcomes)
    neg_stats = c_stats - out_stats
    _, n_cnt = get_neg(claims, outcomes)

    print("-" * 40)
    print(
        f"{type:^9} | {c_cnt:^9} | {out_cnt:^9} | {n_cnt:^9}")

    return [c_stats, out_stats, neg_stats]

def process_dataset(pretokenized_dir, tokenizer, max_len, dataset):

    mlb = MultiLabelBinarizer()
    dataset['train']['claims'], dataset['train']['outcomes'] = binarizer(dataset['train']['claims'], dataset['train']['outcomes'], mlb, True)
    dataset['test']['claims'], dataset['test']['outcomes'] = binarizer(dataset['test']['claims'], dataset['test']['outcomes'], mlb)
    dataset['dev']['claims'], dataset['dev']['outcomes'] = binarizer(dataset['dev']['claims'], dataset['dev']['outcomes'], mlb)

    print(
        f"{'split':^9} | {'claims':^9} | {'positives':^9} | {'negatives':^9}")
    training = data_stats(dataset['train']['claims'], dataset['train']['outcomes'], "train")
    validation = data_stats(dataset['test']['claims'], dataset['test']['outcomes'], "test")
    test = data_stats(dataset['dev']['claims'], dataset['dev']['outcomes'], "dev")

    for i in [training, validation, test]:
        for j in i:
            print(j)

    print('Tokenizing data...')

    Path(pretokenized_dir).mkdir(parents=True, exist_ok=True)

    for i in ['train', 'dev', 'test']:

        facts, masks = preprocessing_for_bert(dataset[i]['facts'], tokenizer, max=max_len)
        arguments, masks_arguments = preprocessing_for_bert(dataset[i]['arguments'], tokenizer, max=max_len)

        with open(pretokenized_dir + "/tokenized_"+i+".pkl", "wb") as f:
            pickle.dump([facts, masks, arguments, masks_arguments, dataset[i]['ids'], dataset[i]['claims'], dataset[i]['outcomes'], dataset[i]['precedent'], mlb], f, protocol=4)


def binarizer(claims, outcomes, mlb, fit=False):
    if fit:
        claims = mlb.fit_transform(claims)
        outcomes = mlb.transform(outcomes)
    else:
        claims = mlb.transform(claims)
        outcomes = mlb.transform(outcomes)

    return claims, outcomes

def preprocessing_for_bert(data, tokenizer, max=512):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """

    # For every sentence...
    input_ids = []
    attention_masks = []

    for sent in tqdm(data):
        sent = " ".join(sent)
        sent = sent[:500000] # Speeds the process up for documents with a lot of precedent we would truncate anyway.
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=max,  # Max length to truncate/pad
            pad_to_max_length=True,  # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,  # Return attention mask
            truncation=True,
        )

        # Add the outputs to the lists
        input_ids.append([encoded_sent.get('input_ids')])
        attention_masks.append([encoded_sent.get('attention_mask')])

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def run():
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    get_data("./datasets/precedent/legal_bert", tokenizer, 512)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    get_data("./datasets/precedent/bert", tokenizer, 512)
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    get_data("./datasets/precedent/longformer", tokenizer, 4096)


def get_allowed(arts):
    allowed = ['10', '11', '13', '14', '18', '2', '3', '4', '5', '6', '7', '8', '9', 'P1-1', 'P4-2', 'P7-1', 'P7-4']
    new = []
    for i in arts:
        if i in allowed:
            new.append(i)
    return new


if __name__ == '__main__':
    run()