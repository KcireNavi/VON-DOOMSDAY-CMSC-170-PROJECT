# AI Disclosure:
# GEMINI 3 PRO
# Used Gemini 3 Pro in this script where it helped in
# connecting all the models into run_comparison() so they run on the same data
# and save their scores in the same format for the table and graphs.

import os
import random
import matplotlib.pyplot as plt
import numpy as np

from preprocess import (
    read_projects,
    split_by_maturity,
    SNAPSHOT_DATE,
    MATURITY_DAYS,
    fit_parameters,
    build_features,
)
from splitting import accuracy, countmetrics
import knn
import naive_bayes
from naive_bayes import run_naive_bayes
from decision_trees import build_tree, predict


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(CODE_DIR, "..", "datasets", "landslides.csv")

SEED = 42
MAX_EVENTS = 6000          # subsample cap so from-scratch KNN finishes in reasonable time
FALSE_POSITIVE_COST = 1
FALSE_NEGATIVE_COST = 2

NB_K_VAL = 0.1
KNN_P = 2
NB_CUTOFFS = [0.3, 0.5, 0.7]
KNN_K_VALUES = [3, 5, 7, 11]
TREE_DEPTHS = [3, 5, 7]

# Positive class = fatal landslide (fatality_count > 0)
INTERPRETATIONS = [
    ("Precision", "Of all landslides flagged as fatal, the share that were truly fatal, showing how much response effort goes to real threats instead of false alarms."),
    ("Recall",    "Of all truly fatal landslides, the share the system flags, showing how many deadly events are caught for early assessment."),
    ("TP",        "A fatal landslide correctly flagged, so response resources can be prioritized early."),
    ("TN",        "A non-fatal landslide correctly left unflagged, so no resources are diverted unnecessarily."),
    ("FP",        "A non-fatal landslide flagged as fatal, causing a false alarm and wasted resources."),
    ("FN",        "A fatal landslide the system missed, the costliest error because a deadly event gets no priority response."),
]


# Creates validation folds (uses its own random generator, so global state is untouched).
def create_folds(projects, k_folds=3, seed=SEED):
    rng = random.Random(seed)
    shuffled = list(projects)
    rng.shuffle(shuffled)

    fold_size = len(shuffled) // k_folds
    folds = []
    for i in range(k_folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < k_folds - 1 else len(shuffled)
        test_set = shuffled[test_start:test_end]
        train_set = shuffled[:test_start] + shuffled[test_end:]
        folds.append((train_set, test_set))

    return folds


# Builds features for one fold.
def prepare_fold(train_projects, test_projects):
    parameters = fit_parameters(train_projects)

    train_features, feature_names, _ = build_features(
        train_projects, parameters, is_train=True)
    test_features, _, _ = build_features(
        test_projects, parameters, is_train=False)

    train_labels = np.array([project["label"] for project in train_projects])
    test_labels = np.array([project["label"] for project in test_projects])

    return train_features, test_features, train_labels, test_labels, feature_names


# Copies positive examples to balance the decision-tree training data.
def balance_data(features, labels):
    minority = features[labels == 1]
    majority = features[labels == 0]

    if len(minority) == 0:
        return features, labels

    multiplier = int(np.ceil(len(majority) / len(minority)))
    balanced_minority = np.vstack([minority] * multiplier)[:len(majority)]

    balanced_features = np.vstack((majority, balanced_minority))
    balanced_labels = np.concatenate((
        np.zeros(len(majority)),
        np.ones(len(balanced_minority)),
    ))
    return balanced_features, balanced_labels


# Calculates metrics, cost, and the confusion matrix.
def evaluate(name, real_labels, predictions):
    precision, recall, f1, confusion = countmetrics(real_labels, predictions)
    fp = confusion[0][1]
    fn = confusion[1][0]

    return {
        "name": name,
        "accuracy": accuracy(real_labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cost": fp * FALSE_POSITIVE_COST + fn * FALSE_NEGATIVE_COST,
        "confusion": confusion,
    }


# Chooses settings by cost on the validation portion.
def choose_settings(train_data, train_labels, test_data,
                    test_labels, feature_names):
    true_labels = test_labels.tolist()

    # Naive Bayes: cutoff
    nb_cutoff, nb_cost = NB_CUTOFFS[0], None
    for cutoff in NB_CUTOFFS:
        predictions = naive_bayes.run_naive_bayes(
            train_data, train_labels.tolist(), test_data,
            feature_names, k_val=NB_K_VAL, cutoff=cutoff)
        cost = evaluate("NB", true_labels, predictions)["cost"]
        if nb_cost is None or cost < nb_cost:
            nb_cutoff, nb_cost = cutoff, cost

    # KNN: k. Distances are measured once and reused for every k.
    train_labels_list = train_labels.tolist()
    class_counts = knn.computeClassCounts(train_labels_list)
    if class_counts[1] > 0:
        fatal_weight = class_counts[0] / class_counts[1]
    else:
        fatal_weight = 1

    all_sorted = [
        knn.getSortedDistances(row, train_data, train_labels_list, KNN_P)
        for row in test_data
    ]

    knn_k, knn_cost = KNN_K_VALUES[0], None
    for k in KNN_K_VALUES:
        predictions = knn.classify(all_sorted, k, fatal_weight)
        cost = evaluate("KNN", true_labels, predictions)["cost"]
        if knn_cost is None or cost < knn_cost:
            knn_k, knn_cost = k, cost

    # Decision Tree: depth
    tree_depth, tree_cost = TREE_DEPTHS[0], None
    balanced_features, balanced_labels = balance_data(train_data, train_labels)
    for depth in TREE_DEPTHS:
        tree = build_tree(
            balanced_features, balanced_labels,
            max_depth=depth,
            min_samples_split=10,
            min_samples_leaf=5)
        predictions = predict(tree, test_data)
        cost = evaluate("Tree", true_labels, predictions)["cost"]
        if tree_cost is None or cost < tree_cost:
            tree_depth, tree_cost = depth, cost

    return nb_cutoff, knn_k, tree_depth


# Averages the metrics over folds and sums cost and confusion counts.
# (Standard deviations are kept only for the graph's error bars.)
def combine_results(name, fold_results):
    combined = {"name": name, "confusion": [[0, 0], [0, 0]]}

    for metric in ["accuracy", "precision", "recall", "f1"]:
        values = [result[metric] for result in fold_results]
        combined[metric] = float(np.mean(values))
        combined[metric + "_std"] = float(np.std(values))

    combined["cost"] = sum(result["cost"] for result in fold_results)

    for result in fold_results:
        for row in range(2):
            for column in range(2):
                combined["confusion"][row][column] += (
                    result["confusion"][row][column])

    return combined


# Prints one comparison table plus a short guide to reading it.
def print_results(results):
    header = (
        f"{'Method':<28}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}"
        f"{'F1':>9}{'Cost':>8}{'TP':>7}{'TN':>7}{'FP':>7}{'FN':>7}"
    )
    line = "-" * len(header)

    print(line)
    print(header)
    print(line)
    for result in results:
        tn, fp = result["confusion"][0]
        fn, tp = result["confusion"][1]
        print(
            f"{result['name']:<28}"
            f"{result['accuracy']:>10.4f}"
            f"{result['precision']:>11.4f}"
            f"{result['recall']:>9.4f}"
            f"{result['f1']:>9.4f}"
            f"{result['cost']:>8.0f}"
            f"{tp:>7}{tn:>7}{fp:>7}{fn:>7}"
        )
    print(line)

    print("Accuracy, Precision, Recall, F1: mean over folds.")
    print("Cost, TP, TN, FP, FN: summed over folds.")
    print(f"Cost = FP x {FALSE_POSITIVE_COST} + FN x {FALSE_NEGATIVE_COST} (lower is better).")

    print("\nHow to read the metrics (positive = fatal landslide):")
    for term, meaning in INTERPRETATIONS:
        print(f"  {term + ':':<10}{meaning}")
    print()


# Creates the bar graph with standard-deviation error bars.
def create_graph(results):
    names = [result["name"] for result in results]
    positions = range(len(names))

    plt.figure(figsize=(12, 6))
    for index, (metric, label) in enumerate([
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1-score")]):
        values = [result[metric] for result in results]
        stds = [result[metric + "_std"] for result in results]
        plt.bar(
            [position + (index - 1.5) * 0.2 for position in positions],
            values,
            0.2,
            yerr=stds,
            capsize=3,
            label=label)

    plt.xticks(list(range(len(names))), names, rotation=20, ha="right")
    plt.xlabel("Method")
    plt.ylabel("Score")
    plt.title("Comparison of Classification Methods (NASA Landslide Catalog)")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(CODE_DIR, "method_comparison.png")
    plt.savefig(path)
    plt.close()
    return path


# Runs every method on all folds and combines the results.
def run_comparison():
    projects = read_projects(CSV_PATH)
    matured_projects, _, _ = split_by_maturity(
        projects, SNAPSHOT_DATE, MATURITY_DAYS)

    if len(matured_projects) > MAX_EVENTS:
        matured_projects = random.Random(SEED).sample(matured_projects, MAX_EVENTS)

    folds = create_folds(matured_projects, k_folds=3)

    # Naive Bayes must bin the landslide numeric columns, not the old budget columns
    naive_bayes.continuous_features = ["latitude", "longitude", "month", "logpop"]

    names = [
        "Always Non-Fatal (Baseline)",
        "Naive Bayes (validated)",
        "KNN Weighted (validated)",
        "Decision Tree (balanced)",
    ]
    fold_results = {name: [] for name in names}

    print(f"Events used: {len(matured_projects)} | 3-fold cross-validation | positive class = fatal")
    print(f"Cost: FP = {FALSE_POSITIVE_COST}, FN = {FALSE_NEGATIVE_COST}\n")

    for fold_number, (train_projects, test_projects) in enumerate(folds, start=1):
        # Settings are chosen on an inner validation split of the training part only
        split_at = int(len(train_projects) * 0.80)
        validation_data = prepare_fold(
            train_projects[:split_at],
            train_projects[split_at:])
        nb_cutoff, knn_k, tree_depth = choose_settings(
            validation_data[0],
            validation_data[2],
            validation_data[1],
            validation_data[3],
            validation_data[4])

        train_data, test_data, train_labels, test_labels, feature_names = (
            prepare_fold(train_projects, test_projects))

        print(
            f"Fold {fold_number}: train={len(train_data)}, test={len(test_data)}, "
            f"fatal in test={int(np.sum(test_labels))} | "
            f"selected: NB cutoff={nb_cutoff}, KNN k={knn_k}, tree depth={tree_depth}")

        predictions = {
            names[0]: [0] * len(test_data),
            names[1]: run_naive_bayes(
                train_data, train_labels.tolist(), test_data,
                feature_names, k_val=NB_K_VAL, cutoff=nb_cutoff),
            names[2]: knn.run_knn(
                train_data, train_labels, test_data,
                k=knn_k, p_val=KNN_P, weighted=True),
        }

        balanced_features, balanced_labels = balance_data(train_data, train_labels)
        tree = build_tree(
            balanced_features, balanced_labels,
            max_depth=tree_depth,
            min_samples_split=10,
            min_samples_leaf=5)
        predictions[names[3]] = predict(tree, test_data)

        for name, predicted_labels in predictions.items():
            fold_results[name].append(evaluate(
                name, test_labels.tolist(), predicted_labels))

    print()
    results = [combine_results(name, fold_results[name]) for name in names]
    print_results(results)

    graph_path = create_graph(results)
    print("Graph saved to: " + graph_path)
    return results


if __name__ == "__main__":
    run_comparison()