import numpy as np
from baselines import baseline_linear
from sklearn.model_selection import train_test_split
import torch
import random


def scores(precs, both):
    # if random:
    #     precs = np.random.randint(2, size=len(precs))

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
            max_recall = recall
            max_precision = precision
            # print("\t".join(map(str, [i, inf, precision, recall, new_f1])))

    return max_F1, max_acc, max_recall, max_precision

def exact_mc_perm_test(x, y, truth, nmc):
    # x = np.array(x)[:, divider:]
    # y = np.array(y)[:, divider:]
    # truth = np.array(truth)[:, divider:]

    """
    There is no difference in f1-micro score between the two models
    """

    n, k = len(x), 0

    F1_1, acc_1, recall_1, precision_1 = scores(truth, x)
    F1_2, acc_2, recall_2, precision_2 = scores(truth, y)

    for S1, S2, name in zip([F1_1, acc_1, recall_1, precision_1], [F1_2, acc_2, recall_2, precision_2], ["f1", "accuracy", "recall", "precission"]):
        diff = np.abs(S1 - S2)
        all_k = 0
        significant = False
        print(name, S1, S2, diff)

        for i in range(nmc):
            switches = np.random.randint(0, 2, n)

            xs_new = []
            ys_new = []
            for a, b, c in zip(x, y, switches):
                if c == 1:
                    xs_new.append(b)
                    ys_new.append(a)
                else:
                    xs_new.append(a)
                    ys_new.append(b)
            if name == "f1":
                s1, _, _, _ = scores(truth, xs_new)
                s2, _, _, _ = scores(truth, ys_new)
            elif name == "precission":
                _, _, _, s1 = scores(truth, xs_new)
                _, _, _, s2 = scores(truth, ys_new)
            elif name == "recall":
                _, _, s1, _ = scores(truth, xs_new)
                _, _, s2, _ = scores(truth, ys_new)
            else:
                _, s1, _, _ = scores(truth, xs_new)
                _, s2, _, _ = scores(truth, ys_new)

            new = np.abs(s1 - s2)
            test = (diff <= new).sum()
            all_k += test
        if (all_k/nmc) < 0.05:
            significant = True
        print(all_k/nmc, significant)
    print("\nEND")
    return all_k / nmc

if __name__ == '__main__':
    tokenized_dir = "../datasets/" + 'precedent' + "/" + 'bert'
    result_path = './outdir/bert/facts/influence_results_tmp_0_False_last-i_294.json'
    batch_size = 2
    all_influences, all_labels = baseline_linear(tokenized_dir, result_path)
    X_train, X_test, y_train, y_test = train_test_split(torch.tensor(all_influences).flatten(0),
                                                        torch.tensor(all_labels).flatten(0),
                                                        test_size=0.33, random_state=42)

    infs = np.array([-float(x) for x in X_train.tolist()])
    truth = np.array(y_train.tolist())
    predicted = sorted(zip(infs, truth), key=lambda x: x[0])
    pred_rand = random.sample(predicted, len(predicted))

    exact_mc_perm_test(predicted, pred_rand, truth, 100)