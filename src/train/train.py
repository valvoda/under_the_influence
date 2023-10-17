import sys
sys.path.append("../../")

from models.bert_classifier import BertClassifier, JointClassifier
# from models.bert_classifier import TestClassifier
from logger import Logger
from src.preprocess.data_loader import DataPrep
from src.preprocess.test_dataset import TestData
# from process_results import get_best

from transformers import AdamW, get_linear_schedule_with_warmup, BertModel, AutoModel, LongformerModel
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

import random
import time
import argparse

import torch
import torch.nn as nn
import numpy as np


class Classifier:

    def __init__(self, model, args):
        self.args = args
        self.device = self.set_cuda()

        if args.arch == "classifier":
            self.model = BertClassifier(model, args, self.device)
        elif args.arch == "joint":
            self.model = JointClassifier(model, args, self.device)

        self.optimizer = AdamW(self.model.parameters(),
                          lr=args.learning_rate, # lr=5e-5,    # Default learning rate
                          eps=1e-8    # Default epsilon value
                          )
        # Set up the learning rate scheduler
        self.scheduler = None

        if args.arch == "classifier":
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        elif args.arch == "joint":
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')

        # MULTI GPU support:
        if self.device == "cuda" and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        self.set_seed()
        self.log = Logger(args)

    def set_cuda(self):
        if torch.cuda.is_available():
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            return torch.device("cuda")
        else:
            print('No GPU available, using the CPU instead.')
            return torch.device("cpu")

    def set_seed(self, seed_value=42):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    def run_epoch(self, dataloader, epoch_i, eval):

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0
        all_truths, all_preds = [], []

        # Put the model into the training/validation mode
        if eval == 'val' or eval == 'test':
            self.model.eval()
        else:
            self.model.train()

        # For each batch of training data...
        for step, batch in enumerate(dataloader):
            batch_counts += 1
            # Load batch to GPU

            b_input_ids, b_attn_mask, b_labels, b_claims = tuple(t.to(self.device) for t in batch)

            # Zero out any previously calculated gradients
            self.model.zero_grad()

            # Perform a forward pass. This will return logits.
            b_input_ids = b_input_ids.squeeze(1)
            b_attn_mask = b_attn_mask.squeeze(1)

            logits = self.model(input_ids=b_input_ids, attention_mask=b_attn_mask)

            if self.args.arch == 'joint':
                loss = self.loss_fn(logits.reshape(-1, 3), b_labels.reshape(-1))
            else:
                loss = self.loss_fn(logits, b_labels.float())

            loss = torch.mean(loss)

            batch_loss += loss.detach().item()
            total_loss += loss.detach().item()

            if self.args.arch == 'joint':
                preds = logits.reshape(b_input_ids.shape[0], -1, 3).argmax(-1)
            else:
                preds = torch.round(torch.sigmoid(logits))

            all_truths += b_labels.cpu().float().tolist()
            all_preds += preds.cpu().float().tolist()

            if eval != 'val' and eval != 'test':
                # Perform a backward pass to calculate gradients
                loss.backward()
                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and the learning rate
                self.optimizer.step()
                self.scheduler.step()
                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (step == len(dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    print(
                        f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {'-':^9} | {'-':^9} | {'-':^9} | {time_elapsed:^9.2f}")

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

        results = {}
        if self.args.arch == 'joint':
            results[eval + "_accuracy"] = accuracy_score(np.array(all_truths) == 1,
                                  np.array(all_preds) == 1) * 100
            results[eval + "_f1"] = f1_score(np.array(all_truths) == 1,
                                  np.array(all_preds) == 1, average="micro") * 100
            results[eval + "_precision"] = precision_score(np.array(all_truths) == 1,
                                  np.array(all_preds) == 1, average="micro") * 100
            results[eval + "_recall"] = recall_score(np.array(all_truths) == 1,
                                  np.array(all_preds) == 1, average="micro") * 100
        else:
            results[eval+"_accuracy"] = accuracy_score(all_truths, all_preds) * 100
            results[eval+"_f1"] = f1_score(all_truths, all_preds, average="micro") * 100
            results[eval+"_precision"] = precision_score(all_truths, all_preds, average="micro") * 100
            results[eval+"_recall"] = recall_score(all_truths, all_preds, average="micro") * 100
        results[eval+"_loss"] = total_loss / len(dataloader)
        outputs = {}
        outputs['truths'] = all_truths
        outputs['preds'] = all_preds

        return results, outputs

    def initialise_scheduler(self, train_dataloader):
        # Total number of training steps
        total_steps = len(train_dataloader) * self.args.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                        num_warmup_steps=0,  # Default value
                                        num_training_steps=total_steps)

    def print_results(self, epoch_i, train, val, test=None):
        # Print performance over the entire training data
        print("-" * 105)
        print(
            f"{epoch_i + 1:^7} | {'nan':^7} | {train['train_loss']:^12.6f} | {val['val_loss']:^10.6f} | {val['val_accuracy']:^9.2f} | "
            f"{val['val_f1']:^9.2f} | {val['val_precision']:^9.2f} | {val['val_recall']:^9.2f} | ")
        print("-" * 105)
        print("\n")

    def train(self, train_dataloader, val_dataloader, test_dataloader):
        """
        Train the model.
        """

        print("Start training...\n")

        self.initialise_scheduler(train_dataloader)

        stop_cnt = 0
        best_loss = 100

        for epoch_i in range(self.args.epochs):
            print(
                f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Val F1':^9} | {'Val Prec':^9} | {'Val Rec':^9} | {'Elapsed':^9}")
            print("-" * 105)

            train, _ = self.run_epoch(train_dataloader, epoch_i, eval='train')

            # run validation
            with torch.no_grad():
                val, _ = self.run_epoch(val_dataloader, epoch_i, eval='val')

            self.print_results(epoch_i, train, val)

            if val['val_loss'] < best_loss:
                best_loss = val['val_loss']
                stop_cnt = 0
                self.log.save_model(self.model)
            else:
                stop_cnt += 1
                print(f"No Improvement! Stop cnt {stop_cnt}")

            if stop_cnt == 1:
                print(f"Early Stopping at {stop_cnt}")
                self.model = self.log.load_model()
                break

        print("Training complete!")

        with torch.no_grad():
            val, _ = self.run_epoch(val_dataloader, epoch_i, eval='val')
            test, outputs = self.run_epoch(test_dataloader, epoch_i, eval='test')

        self.log.save_results({**val, **test}, outputs)

    def run(self, tokenized_dir=None, test=False):
        loader = DataPrep(tokenized_dir, test, self.log, self.args.max_len, self.args.batch_size, self.args.arch, self.args.input)
        # loader = TestData(self.args.batch_size, self.log)
        train_dataloader, val_dataloader, test_dataloader = loader.load()
        self.train(train_dataloader, val_dataloader, test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=512, required=False)
    parser.add_argument("--batch_size", type=int, default=16, required=False)
    parser.add_argument("--learning_rate", type=float, default=3e-5, required=False)
    parser.add_argument("--dropout", type=float, default=0.2, required=False)
    parser.add_argument("--n_hidden", type=int, default=50, required=False)
    parser.add_argument("--epochs", type=int, default=10, required=False)
    parser.add_argument("--model", type=str, default="bert", required=False)
    parser.add_argument("--arch", type=str, default="classifier", required=False)
    parser.add_argument("--dataset", type=str, default="precedent", required=False) # precedent, alleged
    parser.add_argument("--input", type=str, default="facts", required=False) # arguments
    parser.add_argument("--test", dest='test', action='store_true')
    parser.add_argument("--out_dim", type=int, default=14, required=False)


    args = parser.parse_args()
    print(args)

    if args.model == "bert":
        model = BertModel.from_pretrained('bert-base-uncased', gradient_checkpointing=True, return_dict=True)
    elif args.model == "legal_bert":
        model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased", return_dict=True)
    elif args.model == "longformer":
        model = LongformerModel.from_pretrained('allenai/longformer-base-4096', gradient_checkpointing=True,
                                                return_dict=True)
        args.max_length = 4096
    else:
        print("Error: Unsupported Model")

    tokenized_dir = "../datasets/" + args.dataset + "/" + args.model

    cl = Classifier(model, args)
    cl.run(tokenized_dir=tokenized_dir, test=args.test)