# AI Disclosure:
# GEMINI 3 PRO
# Used Gemini 3 Pro in this script where it helped in
# implementing the hardest parts like the Lazy Guesser baseline
# and the class-weighted KNN workaround.

import numpy as np
from preprocess import preprocess
from splitting import accuracy, countmetrics

csv_path = "../datasets/landslides.csv"
k_values = [1, 3, 5, 7, 11, 15] # k values given for observation
p = 2 # Minkowski distance parameter (p = 2 is Euclidean distance)


# counts the amount of non-fatal (0) and fatal (1) events in the training set
def computeClassCounts(train_labels):
    class_counts = {0: 0, 1: 0}

    for label in train_labels:
        class_counts[label] = class_counts[label] + 1

    return class_counts


# measures the Minkowski distance from one test event to every training event
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


# predicts if an event is fatal (1) or non-fatal (0) using the k nearest neighbors
# (same voting logic as Exercise 06, with an optional weight for the rarer fatal class)
def predict(sorted_distances, k, fatal_weight):
    # keep only the k nearest training events
    nearest = sorted_distances[:k]

    # each nearest event votes for its own label
    fatal_votes = 0
    non_fatal_votes = 0
    for n in nearest:
        if n[1] == 1:
            fatal_votes = fatal_votes + fatal_weight
        else:
            non_fatal_votes = non_fatal_votes + 1

    # the label with more votes wins
    if fatal_votes > non_fatal_votes:
        return 1
    elif non_fatal_votes > fatal_votes:
        return 0
    else:
        return nearest[0][1] # if tied, use the label of the closest event


# classifies all test events for a given k and fatal_weight
def classify(all_sorted_distances, k, fatal_weight):
    predictions = []

    for i in range(len(all_sorted_distances)):
        prediction = predict(all_sorted_distances[i], k, fatal_weight)
        predictions.append(prediction)

    return predictions


# Milestone 3 wrapper: takes train data + test features, returns a list of 0s and 1s
def run_knn(train_data, train_labels, test_data, k=15, p_val=2, weighted=True):
    train_labels_list = list(train_labels)
    class_counts = computeClassCounts(train_labels_list)

    if weighted:
        fatal_weight = class_counts[0] / class_counts[1]
    else:
        fatal_weight = 1

    all_sorted_distances = []
    for i in range(len(test_data)):
        sorted_distances = getSortedDistances(test_data[i], train_data, train_labels_list, p_val)
        all_sorted_distances.append(sorted_distances)

    predictions = classify(all_sorted_distances, k, fatal_weight)
    return predictions


def main():
    data = preprocess(csv_path)

    train_data = data["X_train"]
    test_data = data["X_test"]
    train_labels = data["y_train"].tolist()
    test_labels = data["y_test"].tolist()

    class_counts = computeClassCounts(train_labels)

    test_fatal = 0
    for label in test_labels:
        if label == 1:
            test_fatal = test_fatal + 1

    print("Number of events used: " + str(len(train_data) + len(test_data)))
    print("Train: " + str(len(train_data)) + " Test: " + str(len(test_data)))
    print("Train Non-Fatal: " + str(class_counts[0]) + " Fatal: " + str(class_counts[1]))
    print("Test Fatal: " + str(test_fatal) + "\n")

    # Lazy Guesser baseline (same as in naive_bayes.py: always guesses 0 = non-fatal)
    lazy_guesser_labels = []
    for i in range(len(test_labels)):
        lazy_guesser_labels.append(0)

    lazy_guesser_precision, lazy_guesser_recall, lazy_guesser_f1, lazy_guesser_confusion = countmetrics(test_labels, lazy_guesser_labels)

    lazy_guesser_tn = lazy_guesser_confusion[0][0]
    lazy_guesser_fp = lazy_guesser_confusion[0][1]
    lazy_guesser_fn = lazy_guesser_confusion[1][0]
    lazy_guesser_tp = lazy_guesser_confusion[1][1]

    print("Lazy Guesser (always non-fatal)")
    print("TP: " + str(lazy_guesser_tp) + " TN: " + str(lazy_guesser_tn) + " FP: " + str(lazy_guesser_fp) + " FN: " + str(lazy_guesser_fn))
    print("Accuracy: " + str(accuracy(test_labels, lazy_guesser_labels)))
    print("Precision: " + str(lazy_guesser_precision))
    print("Recall: " + str(lazy_guesser_recall))
    print("F1: " + str(lazy_guesser_f1) + "\n")

    # Measure and sort distances from each test event to all training events once
    all_sorted_distances = []
    for i in range(len(test_data)):
        sorted_distances = getSortedDistances(test_data[i], train_data, train_labels, p)
        all_sorted_distances.append(sorted_distances)

    # 1. Standard KNN: every neighbor gets 1 vote (fatal_weight = 1)
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

    # 2. Weighted KNN Workaround: because fatal events are the minority class in train,
    # each fatal neighbor gets (non-fatal count / fatal count) votes so it is not drowned out by the majority class
    fatal_weight = class_counts[0] / class_counts[1]
    print("=== Class-Weighted KNN Workaround (Fatal Vote Weight = " + str(round(fatal_weight, 2)) + ") ===")
    for k in k_values:
        predicted_labels = classify(all_sorted_distances, k, fatal_weight)

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

if __name__ == "__main__":
    main()
    