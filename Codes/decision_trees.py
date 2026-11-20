import numpy as np
from preprocess import preprocess
from splitting import accuracy, countmetrics


def gini(labels):
    finished = np.sum(labels == 0)
    unfinished = np.sum(labels == 1)
    total = len(labels)

    if total == 0:
        return 0

    finished_fraction = finished / total
    unfinished_fraction = unfinished / total
    return 1 - finished_fraction ** 2 - unfinished_fraction ** 2


def main():
    data = preprocess("../datasets/edited_dataset.csv")

    train_features = data["X_train"]
    test_features = data["X_test"]
    train_labels = data["y_train"]
    test_labels = data["y_test"]

    print("Decision Tree - incomplete")


main()