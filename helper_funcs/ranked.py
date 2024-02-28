import numpy as np


def rank5_accuracy(preds, labels):
    rank_1 = 0
    rank_5 = 0

    for (p, gt) in zip(preds, labels):
        # sort descending order
        p = np.argsort(p)[::-1]

        if gt in p[:5]:
            rank_5 += 1

        if gt == p[0]:
            rank_1 += 1

    len_labels = float(len(labels))
    rank_1 /= len_labels
    rank_5 /= len_labels

    return rank_1, rank_5
