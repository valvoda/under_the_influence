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
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.set_seed()
        self.train_loader = None
        self.test_loader = None
        self.X_test = None
        self.y_test = None
        self.get_data()

    def get_data(self):
        tokenized_dir = "../datasets/" + 'precedent' + "/" + 'bert'
        result_path = './outdir/bert/facts/influence_results_tmp_0_False_last-i_294.json'
        batch_size = self.batch_size
        all_influences, all_labels = baseline_linear(tokenized_dir, result_path)
        X_train, X_test, y_train, y_test = train_test_split(torch.tensor(all_influences).flatten(0), torch.tensor(all_labels).flatten(0),
                                                            test_size=0.33, random_state=42)

        self.X_test = X_test
        self.y_test = y_test

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

        iter = 0
        for _ in range(int(self.epochs)):

            self.optimizer.zero_grad()
            self.model.zero_grad()
            self.model.train()

            for i, (influence, labels) in enumerate(self.train_loader):
                outputs = self.model(influence.unsqueeze(0).transpose(0, 1).to(self.device))
                loss = self.criterion(outputs.squeeze(1), labels.float().to(self.device))
                loss.backward()
                self.optimizer.step()

                iter += 1
                if iter % 500 == 0:
                    # calculate Accuracy
                    correct = 0
                    total = 0
                    all_predicted = []
                    all_labels = []
                    for influence, labels in self.test_loader:
                        outputs = self.model(influence.unsqueeze(0).transpose(0, 1))
                        # print(outputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)

                        correct += (predicted == labels).sum()
                        all_predicted += predicted.detach().tolist()
                        all_labels += labels.detach().tolist()
                    accuracy = 100 * correct / total
                    f1 = f1_score(all_labels, all_predicted)
                    print("Iteration: {}. Loss: {}. Accuracy: {}. F1: {}.".format(iter, loss.item(), accuracy, f1))

    def majority_baseline(self):
        labels = [0.0 for _ in self.y_test]
        print("Majority Baseline: Acc: {}. F1 {}.".format(100 *(np.array(self.y_test) == 0.0).mean(), f1_score(labels, self.y_test)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256, required=False)
    parser.add_argument("--lr", type=float, default=0.01, required=False)
    parser.add_argument("--epochs", type=int, default=10, required=False)
    args = parser.parse_args()
    print(args)
    classifier = LinearFit(batch_size=args.batch_size, lr=args.lr, epochs=args.epochs)
    classifier.majority_baseline()
    classifier.train()
    print('done')