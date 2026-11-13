# AI Disclosure:
# GEMINI 3 PRO
# Used Gemini 3 Pro in this script where it helped in
# implementing the hardest parts like the Lazy Guesser baseline
# and the class-weighted KNN workaround, specifically formulating 
# the vote weight of ~55.5 for the minority class to counter the 
# massive difference between the number of "finished" (class 0) and "unfinished" (class 1) projects.

import numpy as np
from preprocess import preprocess
from splitting import accuracy, countmetrics

csv_path = "../datasets/edited_dataset.csv"
k_values = [1, 3, 5, 7, 11, 15] # k values given for observation
p = 2 # Minkowski distance parameter (p = 2 is Euclidean distance)


# counts the amount of finished (0) and unfinished (1) projects in the training set
def computeClassCounts(train_labels):
    class_counts = {0: 0, 1: 0}

    for label in train_labels:
        class_counts[label] = class_counts[label] + 1

    return class_counts


# measures the Minkowski distance from one test project to every training project
# returns a list of (distance, label) pairs sorted from nearest to farthest
def getSortedDistances(point, train_data, train_labels, p):
    # Minkowski formula: ( sum( |a - b|^p ) ) ^ (1/p)
    row_distances = np.sum(np.abs(train_data - point) ** p, axis=1) ** (1 / p)

    distances = []
    for i in range(len(train_data)):
        distances.append((float(row_distances[i]), train_labels[i]))

    # sort from nearest to farthest
    distances.sort()
    return distances


# predicts if a project is unfinished (1) or finished (0) using the k nearest neighbors
# (same voting logic as Exercise 06, with an optional weight for the rare unfinished class)
def predict(sorted_distances, k, unfinished_weight):
    # keep only the k nearest training projects
    nearest = sorted_distances[:k]

    # each nearest project votes for its own label
    unfinished_votes = 0
    finished_votes = 0
    for n in nearest:
        if n[1] == 1:
            unfinished_votes = unfinished_votes + unfinished_weight
        else:
            finished_votes = finished_votes + 1

    # the label with more votes wins
    if unfinished_votes > finished_votes:
        return 1
    elif finished_votes > unfinished_votes:
        return 0
    else:
        return nearest[0][1] # if tied, use the label of the closest project


# classifies all test projects for a given k and unfinished_weight
def classify(all_sorted_distances, k, unfinished_weight):
    predictions = []

    for i in range(len(all_sorted_distances)):
        prediction = predict(all_sorted_distances[i], k, unfinished_weight)
        predictions.append(prediction)

    return predictions


def main():
    data = preprocess(csv_path)

    train_data = data["X_train"]
    test_data = data["X_test"]
    train_labels = data["y_train"].tolist()
    test_labels = data["y_test"].tolist()

    class_counts = computeClassCounts(train_labels)

    test_unfinished = 0
    for label in test_labels:
        if label == 1:
            test_unfinished = test_unfinished + 1

    print("Projects used: " + str(len(train_data) + len(test_data)))
    print("Train: " + str(len(train_data)) + " Test: " + str(len(test_data)))
    print("Train Finished: " + str(class_counts[0]) + " Unfinished: " + str(class_counts[1]))
    print("Test Unfinished: " + str(test_unfinished) + "\n")

    # Lazy Guesser baseline (same as in naive_bayes.py: always guesses 0 = finished)
    lazy_guesser_labels = []
    for i in range(len(test_labels)):
        lazy_guesser_labels.append(0)

    lazy_guesser_precision, lazy_guesser_recall, lazy_guesser_f1, lazy_guesser_confusion = countmetrics(test_labels, lazy_guesser_labels)

    lazy_guesser_tn = lazy_guesser_confusion[0][0]
    lazy_guesser_fp = lazy_guesser_confusion[0][1]
    lazy_guesser_fn = lazy_guesser_confusion[1][0]
    lazy_guesser_tp = lazy_guesser_confusion[1][1]

    print("Lazy Guesser (always finished)")
    print("TP: " + str(lazy_guesser_tp) + " TN: " + str(lazy_guesser_tn) + " FP: " + str(lazy_guesser_fp) + " FN: " + str(lazy_guesser_fn))
    print("Accuracy: " + str(accuracy(test_labels, lazy_guesser_labels)))
    print("Precision: " + str(lazy_guesser_precision))
    print("Recall: " + str(lazy_guesser_recall))
    print("F1: " + str(lazy_guesser_f1) + "\n")

    # Measure and sort distances from each test project to all training projects once
    all_sorted_distances = []
    for i in range(len(test_data)):
        sorted_distances = getSortedDistances(test_data[i], train_data, train_labels, p)
        all_sorted_distances.append(sorted_distances)

    # 1. Standard KNN: every neighbor gets 1 vote (unfinished_weight = 1)
    print("=== Standard KNN (Equal Vote = 1) ===")
    for k in k_values:
        predicted_labels = classify(all_sorted_distances, k, 1)

        precision, recall, f1, confusion = countmetrics(test_labels, predicted_labels)

        tn = confusion[0][0]
        fp = confusion[0][1]
        fn = confusion[1][0]
        tp = confusion[1][1]

        print("k = " + str(k) + " p = " + str(p))
        print("TP: " + str(tp) + " TN: " + str(tn) + " FP: " + str(fp) + " FN: " + str(fn))
        print("Accuracy: " + str(accuracy(test_labels, predicted_labels)))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("F1: " + str(f1) + "\n")

    # 2. Weighted KNN Workaround: because there are 2664 finished vs only 48 unfinished projects in train,
    # each unfinished neighbor gets (2664 / 48 = 55.5) votes so it is not drowned out by the majority class
    unfinished_weight = class_counts[0] / class_counts[1]
    print("=== Class-Weighted KNN Workaround (Unfinished Vote Weight = " + str(round(unfinished_weight, 2)) + ") ===")
    for k in k_values:
        predicted_labels = classify(all_sorted_distances, k, unfinished_weight)

        precision, recall, f1, confusion = countmetrics(test_labels, predicted_labels)

        tn = confusion[0][0]
        fp = confusion[0][1]
        fn = confusion[1][0]
        tp = confusion[1][1]

        print("k = " + str(k) + " p = " + str(p))
        print("TP: " + str(tp) + " TN: " + str(tn) + " FP: " + str(fp) + " FN: " + str(fn))
        print("Accuracy: " + str(accuracy(test_labels, predicted_labels)))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("F1: " + str(f1) + "\n")

main()
