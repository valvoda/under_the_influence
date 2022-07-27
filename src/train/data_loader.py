import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pickle
import numpy as np


class DataPrep:

    def __init__(self, tokenizer_dir, test, log, max_len, batch_size):
        self.tokenized_dir = tokenizer_dir
        self.test = test
        self.log = log
        self.max_len = max_len
        self.batch_size = batch_size

    def make_loader(self, input, mask, labels, claims, train=True):
        labels = torch.tensor(labels)
        claims = torch.tensor(claims)
        data = TensorDataset(input, mask, labels, claims)
        if train:
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        return dataloader

    def load(self):
        with open("../" + self.tokenized_dir + "/tokenized_train.pkl", "rb") as f:
            train_facts, train_masks, train_arguments, \
            train_masks_arguments, train_ids, train_claims, train_outcomes, train_precedent, _ = pickle.load(f)

        with open("../" + self.tokenized_dir + "/tokenized_dev.pkl", "rb") as f:
            val_facts, val_masks, val_arguments, \
            val_masks_arguments, val_ids, val_claims, val_outcomes, val_precedent, _ = pickle.load(f)

        with open("../" + self.tokenized_dir + "/tokenized_test.pkl", "rb") as f:
            test_facts, test_masks, test_arguments, \
            test_masks_arguments, test_ids, test_claims, test_outcomes, test_precedent, _ = pickle.load(f)

        if self.test:
            test_size = 6
            t_size = 2
        else:
            test_size = 100000

        print("Facts in training data")
        train_inputs = train_facts
        train_masks = train_masks
        val_inputs = val_facts
        val_masks = val_masks
        test_inputs = test_facts
        test_masks = test_masks

        train_inputs, train_masks = train_inputs[:test_size, :, :self.max_len], train_masks[:test_size, :, :self.max_len]
        val_inputs, val_masks = val_inputs[:test_size, :, :self.max_len], val_masks[:test_size, :, :self.max_len]
        test_inputs, test_masks = test_inputs[:t_size, :, :self.max_len], test_masks[:t_size, :, :self.max_len]

        pos_train_labels = train_outcomes[:test_size, :]
        pos_val_labels = val_outcomes[:test_size, :]
        pos_test_labels = test_outcomes[:t_size, :]

        train_labels = pos_train_labels
        val_labels = pos_val_labels
        test_labels = pos_test_labels

        claim_train_labels = train_claims[:test_size, :]
        claim_val_labels = val_claims[:test_size, :]
        claim_test_labels = test_claims[:t_size, :]


        self.n_labels = len(train_labels[1])

        # Create the DataLoader for our training set
        train_dataloader = self.make_loader(train_inputs, train_masks, train_labels, claim_train_labels, train=True)
        val_dataloader = self.make_loader(val_inputs, val_masks, val_labels, claim_val_labels, train=False)
        test_dataloader = self.make_loader(test_inputs, test_masks, test_labels, claim_test_labels, train=False)

        if self.log != None:
            self.log.ids = test_ids
            self.log.precedent = test_precedent

        return train_dataloader, val_dataloader, test_dataloader