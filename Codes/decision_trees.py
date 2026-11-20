# AI Disclosure:
# CLAUDE AI SONNET 5.5
# Used CLAUDE AI SONNET 5.5 in decision_trees.py 
# where it helped in implementing building the framework for the decision tree method
# the method to build the tree and the logic for new data points.
# and other requirements given our list of methods

import os
import numpy as np
from preprocess import preprocess
from splitting import accuracy, countmetrics


csv_path = "../datasets/edited_dataset.csv"


# Measures how mixed the finished and unfinished projects are.
def gini(labels):
    if len(labels) == 0:
        return 0.0

    counts = np.bincount(labels.astype(int), minlength=2)
    probabilities = counts / len(labels)

    return 1.0 - np.sum(probabilities ** 2)


def majority_label(labels):
    counts = np.bincount(labels.astype(int), minlength=2)

    if counts[0] >= counts[1]:
        return 0

    return 1


# Finds the feature and threshold that make the best split.
def find_best_split(features, labels, min_samples_leaf):
    parent_impurity = gini(labels)

    best_feature = None
    best_threshold = None
    best_gain = 0.0
    best_left_mask = None
    best_right_mask = None

    for feature_index in range(features.shape[1]):
        values = features[:, feature_index]
        sorted_values = np.sort(np.unique(values))

        if len(sorted_values) < 2:
            continue

        possible_thresholds = (
            sorted_values[:-1] + sorted_values[1:]
        ) / 2.0

        for threshold in possible_thresholds:
            left_mask = values <= threshold
            right_mask = ~left_mask

            left_labels = labels[left_mask]
            right_labels = labels[right_mask]

            if len(left_labels) < min_samples_leaf:
                continue

            if len(right_labels) < min_samples_leaf:
                continue

            weighted_impurity = (
                len(left_labels) * gini(left_labels)
                + len(right_labels) * gini(right_labels)
            ) / len(labels)

            gain = parent_impurity - weighted_impurity

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold
                best_left_mask = left_mask
                best_right_mask = right_mask

    if best_feature is None:
        return None

    return (
        best_feature,
        best_threshold,
        best_left_mask,
        best_right_mask,
        best_gain
    )


# Builds the decision tree by splitting the data into branches.
def build_tree(
    features,
    labels,
    max_depth,
    min_samples_split=10,
    min_samples_leaf=5,
    depth=0
):
    node = {}

    if (
        depth >= max_depth
        or len(labels) < min_samples_split
        or gini(labels) == 0.0
    ):
        node["leaf"] = True
        node["prediction"] = majority_label(labels)
        node["samples"] = len(labels)
        return node

    split = find_best_split(
        features,
        labels,
        min_samples_leaf
    )

    if split is None:
        node["leaf"] = True
        node["prediction"] = majority_label(labels)
        node["samples"] = len(labels)
        return node

    feature_index, threshold, left_mask, right_mask, gain = split

    node["leaf"] = False
    node["feature"] = feature_index
    node["threshold"] = threshold
    node["gain"] = gain
    node["samples"] = len(labels)

    node["left"] = build_tree(
        features[left_mask],
        labels[left_mask],
        max_depth,
        min_samples_split,
        min_samples_leaf,
        depth + 1
    )

    node["right"] = build_tree(
        features[right_mask],
        labels[right_mask],
        max_depth,
        min_samples_split,
        min_samples_leaf,
        depth + 1
    )

    return node


# Follows one project through the tree to make a prediction.
def predict_one(tree, row):
    while not tree["leaf"]:
        if row[tree["feature"]] <= tree["threshold"]:
            tree = tree["left"]
        else:
            tree = tree["right"]

    return tree["prediction"]


# Predicts the labels for all projects in the test set.
def predict(tree, features):
    predictions = []

    for row in features:
        predictions.append(predict_one(tree, row))

    return predictions


# Trains the Decision Tree and returns predictions for the test data.
def run_decision_tree(train_data, train_labels, test_data):
    tree = build_tree(
        train_data,
        np.array(train_labels),
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5
    )

    return predict(tree, test_data)


# Prints the branches and predictions of the completed tree.
def print_tree(tree, feature_names, depth=0):
    prefix = "    " * depth

    if tree["leaf"]:
        print(prefix + "Predict " + str(tree["prediction"]))
        return

    feature_name = feature_names[tree["feature"]]

    print(
        prefix
        + feature_name
        + " <= "
        + str(round(float(tree["threshold"]), 4))
    )

    print(prefix + "Left:")
    print_tree(tree["left"], feature_names, depth + 1)

    print(prefix + "Right:")
    print_tree(tree["right"], feature_names, depth + 1)


# Loads the data, runs the model, and prints its results.
def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    data = preprocess(csv_path)

    train_features = data["X_train"]
    test_features = data["X_test"]
    train_labels = data["y_train"]
    test_labels = data["y_test"]
    feature_names = data["feature_names"]

    test_unfinished = np.sum(test_labels == 1)

    print(
        "Projects used: "
        + str(len(train_features) + len(test_features))
    )
    print(
        "Train: "
        + str(len(train_features))
        + " Test: "
        + str(len(test_features))
    )
    print(
        "Test Unfinished: "
        + str(test_unfinished)
        + "\n"
    )

    lazy_labels = [0] * len(test_labels)

    lazy_precision, lazy_recall, lazy_f1, lazy_confusion = countmetrics(
        test_labels.tolist(),
        lazy_labels
    )

    lazy_tn = lazy_confusion[0][0]
    lazy_fp = lazy_confusion[0][1]
    lazy_fn = lazy_confusion[1][0]
    lazy_tp = lazy_confusion[1][1]

    print("Lazy Guesser (always finished)")
    print(
        "TP:", lazy_tp,
        "TN:", lazy_tn,
        "FP:", lazy_fp,
        "FN:", lazy_fn
    )
    print("Accuracy:", accuracy(test_labels.tolist(), lazy_labels))
    print("Precision:", lazy_precision)
    print("Recall:", lazy_recall)
    print("F1:", lazy_f1)
    print()

    tree = build_tree(
        train_features,
        train_labels,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5
    )

    predicted_labels = predict(tree, test_features)

    precision, recall, f1, confusion = countmetrics(
        test_labels.tolist(),
        predicted_labels
    )

    tn = confusion[0][0]
    fp = confusion[0][1]
    fn = confusion[1][0]
    tp = confusion[1][1]

    print("Decision Tree")
    print("TP:", tp, "TN:", tn, "FP:", fp, "FN:", fn)
    print(
        "Accuracy:",
        accuracy(test_labels.tolist(), predicted_labels)
    )
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    print("\nTree:")
    print_tree(tree, feature_names)


main()