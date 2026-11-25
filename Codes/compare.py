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
from knn import run_knn
import naive_bayes
from naive_bayes import run_naive_bayes
from decision_trees import build_tree, predict


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(CODE_DIR, "..", "datasets", "landslides.csv")

FALSE_POSITIVE_COST = 1
FALSE_NEGATIVE_COST = 2


# Creates validation folds.
def create_folds(projects, k_folds=3, seed=42):
    random.seed(seed)
    shuffled = list(projects)
    random.shuffle(shuffled)

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

    train_features, feature_names, continuous_positions = build_features(
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
    precision, recall, f1, confusion = countmetrics(
        real_labels, predictions)
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
    nb_cutoff = 0.5
    nb_cost = None
    for cutoff in [0.3, 0.5, 0.7]:
        predictions = run_naive_bayes(
            train_data, train_labels.tolist(), test_data,
            feature_names, k_val=0.1, cutoff=cutoff)
        cost = evaluate("NB", test_labels.tolist(), predictions)["cost"]
        if nb_cost is None or cost < nb_cost:
            nb_cutoff, nb_cost = cutoff, cost

    knn_k = 5
    knn_cost = None
    for k in [3, 5, 7, 11]:
        predictions = run_knn(
            train_data, train_labels, test_data,
            k=k, p_val=2, weighted=True)
        cost = evaluate("KNN", test_labels.tolist(), predictions)["cost"]
        if knn_cost is None or cost < knn_cost:
            knn_k, knn_cost = k, cost

    tree_depth = 5
    tree_cost = None
    balanced_features, balanced_labels = balance_data(
        train_data, train_labels)
    for depth in [3, 5, 7]:
        tree = build_tree(
            balanced_features, balanced_labels,
            max_depth=depth,
            min_samples_split=10,
            min_samples_leaf=5)
        predictions = predict(tree, test_data)
        cost = evaluate("Tree", test_labels.tolist(), predictions)["cost"]
        if tree_cost is None or cost < tree_cost:
            tree_depth, tree_cost = depth, cost

    return nb_cutoff, knn_k, tree_depth


# Finds fold averages, standard deviations, and total confusion counts.
def combine_results(name, fold_results):
    combined = {"name": name, "confusion": [[0, 0], [0, 0]]}

    for metric in ["accuracy", "precision", "recall", "f1", "cost"]:
        values = [result[metric] for result in fold_results]
        combined[metric] = float(np.mean(values))
        combined[metric + "_std"] = float(np.std(values))

    for result in fold_results:
        for row in range(2):
            for column in range(2):
                combined["confusion"][row][column] += (
                    result["confusion"][row][column])

    return combined


# Prints mean metrics and combined confusion matrices.
def print_results(results):
    print("Method                       Accuracy            Precision           Recall              F1                  Cost")

    for result in results:
        print(
            f"{result['name']:<28}"
            + f'{result["accuracy"]:.4f} +/- {result["accuracy_std"]:.4f}   '
            + f'{result["precision"]:.4f} +/- {result["precision_std"]:.4f}   '
            + f'{result["recall"]:.4f} +/- {result["recall_std"]:.4f}   '
            + f'{result["f1"]:.4f} +/- {result["f1_std"]:.4f}   '
            + f'{result["cost"]:.1f} +/- {result["cost_std"]:.1f}'
        )

    print("\nCombined Confusion Matrices:")
    for result in results:
        tn, fp = result["confusion"][0]
        fn, tp = result["confusion"][1]
        print(f'{result["name"]}: TP={tp}, TN={tn}, FP={fp}, FN={fn}')


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
    plt.savefig(os.path.join(CODE_DIR, "method_comparison.png"))


# Runs every method on all folds and combines the results.
def run_comparison():
    projects = read_projects(CSV_PATH)
    matured_projects, _, _ = split_by_maturity(
        projects, SNAPSHOT_DATE, MATURITY_DAYS)

    # Subsample if dataset is large so from-scratch KNN completes in reasonable time
    if len(matured_projects) > 6000:
        random.seed(42)
        matured_projects = random.sample(matured_projects, 6000)

    folds = create_folds(matured_projects, k_folds=3)

    # Point naive_bayes continuous_features to the landslide numeric columns
    naive_bayes.continuous_features = ["latitude", "longitude", "month", "logpop"]

    names = [
        "Always Non-Fatal (Baseline)",
        "Naive Bayes (validated)",
        "KNN Weighted (validated)",
        "Decision Tree (balanced)",
    ]
    fold_results = {name: [] for name in names}

    print("Three validation folds")
    print(
        f"Cost: FP={FALSE_POSITIVE_COST}, "
        f"FN={FALSE_NEGATIVE_COST}\n")

    for fold_number, (train_projects, test_projects) in enumerate(
            folds, start=1):
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
            f"Fold {fold_number}: train={len(train_data)}, "
            f"test={len(test_data)}, "
            f"fatal in test={sum(test_labels)}")
        print(
            f"  Selected: cutoff={nb_cutoff}, "
            f"k={knn_k}, depth={tree_depth}")

        predictions = {
            names[0]: [0] * len(test_data),
            names[1]: run_naive_bayes(
                train_data, train_labels.tolist(), test_data,
                feature_names, k_val=0.1, cutoff=nb_cutoff),
            names[2]: run_knn(
                train_data, train_labels, test_data,
                k=knn_k, p_val=2, weighted=True),
        }

        balanced_features, balanced_labels = balance_data(
            train_data, train_labels)
        tree = build_tree(
            balanced_features, balanced_labels,
            max_depth=tree_depth,
            min_samples_split=10,
            min_samples_leaf=5)
        predictions[names[3]] = predict(tree, test_data)

        for name, predicted_labels in predictions.items():
            fold_results[name].append(evaluate(
                name, test_labels.tolist(), predicted_labels))

    print("")
    results = [
        combine_results(name, fold_results[name])
        for name in names
    ]
    print_results(results)
    create_graph(results)
    return results


if __name__ == "__main__":
    run_comparison()
