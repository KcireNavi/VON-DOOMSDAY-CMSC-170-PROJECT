from preprocess import preprocess
from splitting import accuracy, countmetrics

csv_path = "../datasets/edited_dataset.csv"
k = [2.0, 1.0, 0.5, 0.1, 0.005] # k values given for observation

threshold = 0.5
bin_num = 10        # number of categories to split the budgets into
continuous_features = ["log_budget", "log_contractor_train_projects"] # the only columns that are not 0 or 1, so the only ones that get binned


def computeBinEdges(values, bins): # calculates the cutoff points per bin (edges) from the train projects
    sorted_values = sorted(values)
    edges = []

    for i in range(1,bins):
        pos = int(len(sorted_values) * i/bins)  # i/bins of the way through the entire sorted list ( i = 1  bins = 10 becomes 1/10 or 10% of the way through the list)
        edges.append(sorted_values[pos]) # budget of that specific project would be the edge

    return edges


def getBin(budget, edges): # finds the exact bin where a budget would fall into
    bin_number = 0

    for edge in edges:
        if budget >= edge:
            bin_number = bin_number + 1

    return "bin" + str(bin_number)


def rowToBinnedFeatures(row, feature_names, edges): # turns a row of numbers into {feature: category}, conntinuous columns then is assigned bins, and 0 or 1 columns are left untouched
    categories = {}

    for i in range(len(feature_names)):
        feature = str(feature_names[i])

        if feature in edges: # continuous columns
            categories[feature] = getBin(row[i], edges[feature])
        else: # 0 or 1 columns
            categories[feature] = int(row[i])

    return categories


def computeCounts(X_train, train_labels, feature_names, edges): # counts the projects per class and how often every feature value appeared per class (somewhat similar to exer 4 bag of words but just slightly more complicated)
    class_counts = {0: 0, 1:0} # stores amount of finished and unfinished train projects(0 = finished  1 = unfinished)
    value_counts = {0: {}, 1: {}} # stores the amount of times that value of the feature from that class appeared
    values_seen = {} # stores all specific values found throughout both classes for every feature

    for i in range(len(X_train)):
        categories = rowToBinnedFeatures(X_train[i], feature_names, edges)
        label = train_labels[i]
        class_counts[label] = class_counts[label] + 1

        for feature in categories:
            value = categories[feature]

            if feature not in value_counts[label]:
                value_counts[label][feature] = {}

            if feature not in values_seen:
                values_seen[feature] = []

            if value in value_counts[label][feature]:
                value_counts[label][feature][value] = value_counts[label][feature][value] + 1
            else:
                value_counts[label][feature][value] = 1

            if value not in values_seen[feature]:
                values_seen[feature].append(value)

    return class_counts, value_counts, values_seen


def computeValueProb(feature, value, label, class_counts, value_counts, values_seen, k_val):
    count = 0

    if feature in value_counts[label] and value in value_counts[label][feature]:
        count = value_counts[label][feature][value]

    return (count + k_val) / (class_counts[label] + k_val * len(values_seen[feature]))


def computeLikelihood(categories, label, class_counts, value_counts, values_seen, k_val):
    likelihood = 1.0

    for feature in categories:
        value = categories[feature]

        if value not in values_seen[feature]: # ignore a value that was never seen in train (same idea as exer 4 by ignoring unkown words)
            continue

        value_prob = computeValueProb(feature, value, label, class_counts, value_counts, values_seen, k_val)

        likelihood = likelihood * value_prob

    return likelihood


def computeUnfinishedProb(categories, class_counts, value_counts, values_seen, k_val): # (similar structure as computeSpamProbOverMsg in exer 4)
    total = class_counts[0] + class_counts[1]
    unfinished_prob = class_counts[1] / total # P(Unfinished)
    finished_prob = class_counts[0] / total # P(Finished)

    unfinished_likelihood = computeLikelihood(categories, 1, class_counts, value_counts, values_seen, k_val)
    finished_likelihood = computeLikelihood(categories, 0, class_counts, value_counts, values_seen, k_val)

    # P(project) = P(project | Uninished) P(Unfinished) + P(project |Finished) P(Finished)
    project_prob = unfinished_likelihood * unfinished_prob + finished_likelihood * finished_prob

    return (unfinished_likelihood * unfinished_prob) / project_prob


def classify(X_test, feature_names, edges, class_counts, value_counts, values_seen, k_val, cutoff): # classifies if unfinished or finished and the prob for every test project
    labels = []
    probabilities = []

    for project in X_test:
        categories = rowToBinnedFeatures(project, feature_names, edges)
        prob = computeUnfinishedProb(categories, class_counts, value_counts, values_seen, k_val)

        if prob < cutoff:
            labels.append(0)
        else:
            labels.append(1)
        probabilities.append(prob)

    return labels, probabilities


# Milestone 3 wrapper: takes train data + test features, returns a list of 0s and 1s
def run_naive_bayes(train_data, train_labels, test_data, feature_names, k_val=0.1, cutoff=threshold):
    train_labels_list = list(train_labels)

    edges = {}
    for i in range(len(feature_names)):
        feature = str(feature_names[i])
        if feature in continuous_features:
            edges[feature] = computeBinEdges(train_data[:, i], bin_num)

    class_counts, value_counts, values_seen = computeCounts(train_data, train_labels_list, feature_names, edges)
    predicted_labels, probabilities = classify(test_data, feature_names, edges, class_counts, value_counts, values_seen, k_val, cutoff)
    return predicted_labels

        
def main():
    data = preprocess(csv_path)
 
    X_train = data["X_train"]
    X_test = data["X_test"]
    train_labels = data["y_train"].tolist()
    test_labels = data["y_test"].tolist()
    feature_names = data["feature_names"]
 
    edges = {} # edges[feature] = bin cutoffs, only for the continuous columns
    for i in range(len(feature_names)):
        feature = str(feature_names[i])
        if feature in continuous_features:
            edges[feature] = computeBinEdges(X_train[:, i], bin_num)
 
    class_counts, value_counts, values_seen = computeCounts(X_train, train_labels, feature_names, edges)
 
    test_unfinished = 0
    for label in test_labels:
        if label == 1:
            test_unfinished = test_unfinished + 1
 
    print("Projects used: " + str(len(X_train) + len(X_test)))
    print("Train: " + str(len(X_train)) + " Test: " + str(len(X_test)))
    print("Train Finished: " + str(class_counts[0]) + " Unfinished: " + str(class_counts[1]))
    print("Test Unfinished: " + str(test_unfinished) + "\n")
 
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
 
    for k_val in k:
        predicted_labels, probabilities = classify(X_test, feature_names, edges, class_counts, value_counts, values_seen, k_val, threshold)
 
        precision, recall, f1, confusion = countmetrics(test_labels, predicted_labels)
 
        tn = confusion[0][0]
        fp = confusion[0][1]
        fn = confusion[1][0]
        tp = confusion[1][1]
 
        print("k = " + str(k_val) + " threshold = " + str(threshold))
        print("TP: " + str(tp) + " TN: " + str(tn) + " FP: " + str(fp) + " FN: " + str(fn))
        print("Accuracy: " + str(accuracy(test_labels, predicted_labels)))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("F1: " + str(f1) + "\n")

if __name__ == "__main__":
    main()