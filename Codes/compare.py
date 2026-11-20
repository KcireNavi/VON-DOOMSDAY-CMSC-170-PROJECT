# AI Disclosure:
# GEMINI 3 PRO
# Used Gemini 3 Pro in this script where it helped in
# connecting all the models into run_comparison() so they run on the same data
# and save their scores in the same format for the table and graphs.

import os
import matplotlib.pyplot as plt
from preprocess import preprocess
from splitting import accuracy, countmetrics
from knn import run_knn
from naive_bayes import run_naive_bayes
from decision_trees import build_tree, predict

if os.path.exists("datasets/edited_dataset.csv"):
    csv_path = "datasets/edited_dataset.csv"
else:
    csv_path = "../datasets/edited_dataset.csv"


# counts how many labels in a list are unfinished (1)
def count_unfinished(labels):
    count = 0
    for label in labels:
        if label == 1:
            count = count + 1
    return count


# adds spaces to the end of a string so columns line up nicely in the table
def pad_text(text, width):
    result = str(text)
    while len(result) < width:
        result = result + " "
    return result


# Baseline method: always predicts 0 (finished) for every test project
def run_always_finished(train_data, train_labels, test_data):
    predictions = []
    for i in range(len(test_data)):
        predictions.append(0)
    return predictions


# Scores one method's predictions and puts the metrics into a dictionary
def evaluate_method(name, real_labels, predicted_labels):
    acc = accuracy(real_labels, predicted_labels)
    precision, recall, f1, confusion = countmetrics(real_labels, predicted_labels)

    result = {
        "name": name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion, # [[TN, FP], [FN, TP]]
    }
    return result


# Prints a clean comparison table and the confusion matrix breakdown for every method
def print_results_table(results):
    line = "-" * 96
    print(line)
    header = (
        pad_text("Method", 28)
        + " | " + pad_text("Accuracy", 8)
        + " | " + pad_text("Precision", 9)
        + " | " + pad_text("Recall", 8)
        + " | " + pad_text("F1", 8)
        + " | Confusion Matrix"
    )
    print(header)
    print(line)

    for res in results:
        acc_text = str(round(res["accuracy"], 4))
        prec_text = str(round(res["precision"], 4))
        rec_text = str(round(res["recall"], 4))
        f1_text = str(round(res["f1"], 4))
        cm_text = str(res["confusion_matrix"])

        row = (
            pad_text(res["name"], 28)
            + " | " + pad_text(acc_text, 8)
            + " | " + pad_text(prec_text, 9)
            + " | " + pad_text(rec_text, 8)
            + " | " + pad_text(f1_text, 8)
            + " | " + cm_text
        )
        print(row)

    print(line)
    print("")

    # Also print TN, FP, FN, TP clearly for each method
    print("Confusion Matrix Details:")
    for res in results:
        confusion = res["confusion_matrix"]
        tn = confusion[0][0]
        fp = confusion[0][1]
        fn = confusion[1][0]
        tp = confusion[1][1]

        print(res["name"])
        print("  TP: " + str(tp) + "  TN: " + str(tn) + "  FP: " + str(fp) + "  FN: " + str(fn))

    #function for graphing the results of the methods in a bar graph
def create_metrics_graph(results):
    methods = []
    accuracy_values = []
    precision_values = []
    recall_values = []
    f1_values = []

    for result in results:
        methods.append(result["name"])
        accuracy_values.append(result["accuracy"])
        precision_values.append(result["precision"])
        recall_values.append(result["recall"])
        f1_values.append(result["f1"])

    positions = list(range(len(methods)))
    width = 0.2

    plt.figure(figsize=(12, 6))

    plt.bar(
        [x - 1.5 * width for x in positions],
        accuracy_values,
        width,
        label="Accuracy"
    )

    plt.bar(
        [x - 0.5 * width for x in positions],
        precision_values,
        width,
        label="Precision"
    )

    plt.bar(
        [x + 0.5 * width for x in positions],
        recall_values,
        width,
        label="Recall"
    )

    plt.bar(
        [x + 1.5 * width for x in positions],
        f1_values,
        width,
        label="F1-score"
    )

    plt.xlabel("Method")
    plt.ylabel("Score")
    plt.title("Comparison of Classification Methods")
    plt.xticks(positions, methods, rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    plt.savefig("method_comparison.png")
    plt.show()
    
# Runs all methods on the exact same dataset split and returns the list of results
def run_comparison():
    data = preprocess(csv_path)

    train_data = data["X_train"]
    test_data = data["X_test"]
    train_labels = data["y_train"].tolist()
    test_labels = data["y_test"].tolist()
    feature_names = data["feature_names"]

    train_unfinished = count_unfinished(train_labels)
    test_unfinished = count_unfinished(test_labels)

    print("Projects used: " + str(len(train_data) + len(test_data)))
    print("Train: " + str(len(train_data)) + " (Unfinished: " + str(train_unfinished) + ")")
    print("Test: " + str(len(test_data)) + " (Unfinished: " + str(test_unfinished) + ")")
    print("")

    results = []

    # 1. Always Finished Baseline
    baseline_preds = run_always_finished(train_data, train_labels, test_data)
    baseline_result = evaluate_method("Always Finished (Baseline)", test_labels, baseline_preds)
    results.append(baseline_result)

    # 2. Naive Bayes
    nb_preds = run_naive_bayes(train_data, train_labels, test_data, feature_names)
    nb_result = evaluate_method("Naive Bayes (k=0.1)", test_labels, nb_preds)
    results.append(nb_result)

    # 3. KNN
    knn_std_preds = run_knn(train_data, train_labels, test_data, 5, 2, False)
    knn_std_result = evaluate_method("KNN Standard (k=5)", test_labels, knn_std_preds)
    results.append(knn_std_result)

    # 4. Decision Tree
    tree = build_tree(train_data, data["y_train"], 5, 10, 5)
    dt_preds = predict(tree, test_data)
    dt_result = evaluate_method("Decision Tree (depth=5)", test_labels, dt_preds)
    results.append(dt_result)

    # 5. Graph and Table Output
    print_results_table(results)
    create_metrics_graph(results)
    return results


if __name__ == "__main__":
    run_comparison()
