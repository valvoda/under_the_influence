import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
import random
from baselines import baseline_linear
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import argparse

class LogisticRegression(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(input_dim, output_dim)

     def forward(self, x):
         logits = self.linear(x)
         outputs = torch.sigmoid(logits)
         return outputs



class LinearFit:

    def __init__(self, batch_size, lr, epochs):
        D_in, D_out = 1, 1
        self.epochs = epochs
        self.device = self.set_cuda()
        self.batch_size = batch_size
        self.lr = lr
        self.model = LogisticRegression(D_in, D_out).to(self.device)
        self.criterion = nn.BCELoss()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.set_seed()
        self.train_loader = None
        self.test_loader = None
        self.X_test = None
        self.y_test = None
        self.get_data()

    def get_data(self):
        tokenized_dir = "../datasets/" + 'precedent' + "/" + 'bert'
        result_path = './outdir/bert/facts/927927e50ca941ceb7a0b09b51fe54fb_750_1000_0_False_last-i_137.json'
        # result_path = './outdir/bert/facts/influence_results_tmp_0_False_last-i_294.json'
        batch_size = self.batch_size
        all_influences, all_labels = baseline_linear(tokenized_dir, result_path)
        X_train, X_test, y_train, y_test = train_test_split(torch.tensor(all_influences).flatten(0), torch.tensor(all_labels).flatten(0),
                                                            test_size=0.33, random_state=42)

        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train

        self.train_loader = torch.utils.data.DataLoader(dataset=TensorDataset(X_train, y_train), batch_size=batch_size,
                                                   shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=TensorDataset(X_test, y_test), batch_size=batch_size,
                                                  shuffle=False)



    def set_seed(self, seed_value=42):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    def set_cuda(self):
        if torch.cuda.is_available():
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            return torch.device("cuda")
        else:
            print('No GPU available, using the CPU instead.')
            return torch.device("cpu")

    def train(self):

        for e in range(int(self.epochs)):
            # print("Epoch: ", e)
            self.optimizer.zero_grad()
            self.model.zero_grad()
            self.model.train()

            epoch_loss = []

            for i, (influence, labels) in enumerate(self.train_loader):
                in_inf = influence.unsqueeze(0).transpose(0, 1).to(self.device)
                # in_inf = labels.float().unsqueeze(0).transpose(0, 1).to(self.device)
                outputs = self.model(in_inf)
                in_lab = labels.float().to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(outputs.squeeze(1), in_lab)
                epoch_loss.append(loss.detach().to('cpu'))
                loss.backward()
                self.optimizer.step()


            correct = 0
            total = 0
            all_predicted = []
            all_labels = []

            self.model.eval()
            with torch.no_grad():
                for influence, labels in self.test_loader:
                    in_inf = influence.unsqueeze(0).transpose(0, 1).to(self.device)
                    # in_inf = labels.float().unsqueeze(0).transpose(0, 1).to(self.device)
                    outputs = self.model(in_inf)
                    # print(outputs)
                    predicted = outputs.reshape(-1).to('cpu').detach().numpy().round()
                    total += labels.size(0)

                    correct += (predicted == np.array(labels)).sum()
                    all_predicted += list(predicted)
                    all_labels += labels
                accuracy = 100 * correct / total
                f1 = f1_score(all_labels, all_predicted)
                print("Epoch: {}. Train Loss: {}. Test Accuracy: {}. F1: {}.".format(e, np.array(epoch_loss).mean(), accuracy, f1))

    def majority_baseline(self):
        labels = [0.0 for _ in self.y_test]
        print("Majority Baseline: Acc: {}. F1 {}.".format(100 *(np.array(self.y_test) == 0.0).mean(), f1_score(labels, self.y_test)))

    def linear_search(self, random=False):

        infs = np.array([-float(x) for x in self.X_train.tolist()])
        precs = np.array(self.y_train.tolist())

        both = sorted(zip(infs, precs), key=lambda x: x[0])

        if random:
            np.random.shuffle(both)

        # initialization
        Np, p = 0, 0
        Nr, r = 0, 0
        Np = len(precs)
        for prec in precs:
            if prec == 1:
                p += 1
        r = p
        Nr = p

        def f1(p, Np, r, Nr):
            precision = 0
            if int(Np) != 0:
                precision = float(p) / Np
            recall = 0
            if int(Nr) != 0:
                recall = float(r) / Nr
            if int(r) == 0:  # hack for fast
                precision = 1

            return precision, recall, 2 * precision * recall / (precision + recall)

        max_F1 = 0.0
        max_acc = 0.0
        max_precision = 0.0
        max_recall = 0.0

        for i, (inf, prec) in enumerate(both):
            # take prec to be the new boundary
            if prec == 1:
                p -= 1
                r -= 1

            Np -= 1

            precision, recall, new_f1 = f1(p, Np, r, Nr)
            new_acc = p / float(len(precs))
            # print(1.0-new_acc)
            if new_acc > max_acc:
                # print("\t".join(map(str, [i, 1.0 - new_acc])))
                # print('Best acc: ', 1.0-new_acc)
                max_acc = new_acc

            if new_f1 > max_F1:
                max_F1 = new_f1
                # print("\t".join(map(str, [i, inf, precision, recall, new_f1])))

        return max_F1, max_acc, max_recall, max_precision


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256, required=False)
    parser.add_argument("--lr", type=float, default=0.001, required=False)
    parser.add_argument("--epochs", type=int, default=10, required=False)
    args = parser.parse_args()
    print(args)
    classifier = LinearFit(batch_size=args.batch_size, lr=args.lr, epochs=args.epochs)
    # classifier.majority_baseline()
    print('influences:', classifier.linear_search(random=False))
    for i in range(100):
        print(f'{i} random {classifier.linear_search(random=True)}')
    # classifier.train()
    # print('done')