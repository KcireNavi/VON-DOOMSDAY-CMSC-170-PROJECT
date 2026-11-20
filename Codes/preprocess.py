# Preprocessing the DPWH Infrastructure Transparency Dataset
"""
    AI DECLARATION FORM

    [1] Used Claude AI (Sonnet 5.5) to generate majority of the codes 
    within this file in the context of our goal. The purpose of this file 
    is to preprocess the dataset to turn messy data into structured, usable
    format so that the methods can actually learn from it.
"""

import csv
import datetime
import math
import re
import sys
import numpy as np

# DEFINITION OF CONSTANTS

# date the data was collected ("last updated" field from the website)
SNAPSHOT_DATE = datetime.date(2025, 11, 29)

# arbitrarily, a project needs 2 years to have a fair outcome          
MATURITY_DAYS = 2 * 365

# percent of the "matured" projects used for training; whatever is left over (0.30) becomes the test set
TRAIN_FRACTION = 0.70       

# a category must appear at least this many times in the training set to get its own column
# rare categories are merge into one "__OTHER__" column
MIN_COUNT_REGION = 1
MIN_COUNT_PROVINCE = 25

# columns we need from the .csv dataset
DATA_COLUMNS = ["category", "budget", "location", "contractor", "status" ,"startDate"]

# text values that really mean "no value" (blank cells, Excel errors)
MISSING_TEXT = ["", "#ref!", "#n/a", "#value!", "nan", "null", "none"]

# status values that count as finished (label 0)
# other status ("on-going", "terminated") count as not finished (label 1)
FINISHED_STATUS_VALUES = ["completed"]

OTHER_COLUMN_NAME = "__OTHER__"

# PARSING VALUES

# Strip whitespaces; turn blank / error strings into None
def clean(text):
    text = text.lower().strip()
    if text in MISSING_TEXT:
        return None
    
    return text

# Parse a number; return None is missing / not a number
def to_float(text):
    if text is None:
        return None
    try:
        x = float(text)
    except ValueError:
        return None
    if math.isfinite(x):
        return x
    
    return None

# Convert a date written as day/month/year to a date
def parse_date(text):
    if text is None:
        return None

    possible_formats = ["%m/%d/%Y", "%Y-%m-%d"]
    for date_format in possible_formats:
        try:
            parsed = datetime.datetime.strptime(text, date_format)
            return parsed.date()
        except ValueError:
            continue

    return None

# split "{'province': Laguna 2nd DEO, 'region': Region IV-A}" into two parts.
# text is not a valid JSON (names have no qoutes), so a regular expression is used
# returns (province, region); both are None if the text does not match
def parse_location(text):
    if text is None:
        return None, None

    match = re.search(r"'province':\s*(.*?),\s*'region':\s*(.*?)\}", text)
    if match is None:
        return None, None

    province = match.group(1).strip()
    region = match.group(2).strip()

    return province, region

# extract a contractor identity and a 'revoked' flag
# the number in brackets is used as the identity. If no number is found, the upper-cased name is used
# returns (contractor_id, revoked flag)
def parse_contractor(text):
    if text is None:
        return None

    revoked_flag = 0.0
    if "[REVOKED]" in text.upper():
        revoked_flag = 1.0

    match = re.search(r"\((?:\[[^\]]*\]\s*)?(\d+)\)\s*$", text)
    if match is not None:
        contractor_id = match.group(1)
    else: 
        contractor_id = text.upper()

    return contractor_id, revoked_flag

# turn multiple types into list (ex. "Bridges, Roads" -> ["Bridges", "Roads"])
def parse_project_types(text):
    if text is None:
        return []

    types = []
    for type in text.split(","):
        type = type.strip()
        if type != "":
            types.append(type)

    return types

# convert a status text to a label: 0 = finished, 1 = not finished
def status_to_label(status):
    if status.lower() in FINISHED_STATUS_VALUES:
        return 0

    return 1

# READING THE .CSV FILE

# map each column name to the position of its first occurence
def find_column_positions(header):
    positions = {}

    for position in range(len(header)):
        name = header[position].strip()
        if name == "":
            continue
        if name not in positions:
            positions[name] = position

    return positions

def read_cell(row, positions, column_name):
    position = positions[column_name]

    if position >= len(row):
        return None
    
    return clean(row[position])

# read the .csv file and return a list of dictionaries, one per project
def read_projects(path):
    projects = []
 
    with open(path, newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        positions = find_column_positions(header)
 
        for column_name in DATA_COLUMNS:
            if column_name not in positions:
                raise ValueError("Column missing from the CSV: " + column_name)

        # Read the raw (still text) value of every column.
        for row in reader:
            raw = {}
            for column_name in DATA_COLUMNS:
                raw[column_name] = read_cell(row, positions, column_name)
 
            # Skip empty lines.
            row_is_blank = True
            for column_name in DATA_COLUMNS:
                if raw[column_name] is not None:
                    row_is_blank = False
            if row_is_blank:
                continue
 
            # A project without a status has no label, so it cannot be used.
            if raw["status"] is None:
                continue
 
            # The start date is kept only for the maturity rule and the train/test split. Without it, skip the row.
            start_date = parse_date(raw["startDate"])
            if start_date is None:
                continue
 
            province, region = parse_location(raw["location"])
            contractor_id, revoked_flag = parse_contractor(raw["contractor"])
 
            # A budget of 0 (or less) is a placeholder.
            budget = to_float(raw["budget"])
            if budget is not None:
                if budget <= 0:
                    budget = None

            project = {
                "types": parse_project_types(raw["category"]),
                "budget": budget,
                "province": province,
                "region": region,
                "contractor_id": contractor_id,
                "revoked": revoked_flag,
                "label": status_to_label(raw["status"]),
                "start_date": start_date,
            }
            projects.append(project)

    return projects

# HELPER FUNCTIONS

def median(values):
    sorted_values = sorted(values)
    count = len(sorted_values)
    middle = count // 2
 
    if count % 2 == 1:
        return sorted_values[middle]
 
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2.0

# collect one field from every project into a list
def collect_values(projects, field):
    values = []
 
    for project in projects:
        values.append(project[field])
 
    return values

# frequency table {value: number of times it appears}
def count_values(values):
    counts = {}
 
    for value in values:
        if value is None:
            continue
        if value in counts:
            counts[value] = counts[value] + 1
        else:
            counts[value] = 1
 
    return counts

# sort key for (category, count) pairs: highest count first, then A-Z
def sort_key_count_then_name(item):
    category, count = item
    return (-count, category)   # negative count puts big counts first when sorting ascending

# list the categories that appear at least 'min_count' times
def fit_vocabulary(values, min_count):
    counts = count_values(values)
    ordered_items = sorted(counts.items(), key=sort_key_count_then_name)
 
    vocabulary = []
    for category, count in ordered_items:
        if count >= min_count:
            vocabulary.append(category)
 
    return vocabulary

# one-hot encode a list of categories
def one_hot(values, vocabulary):
    position_of = {}
    for position in range(len(vocabulary)):
        position_of[vocabulary[position]] = position
 
    other_position = len(vocabulary)
    matrix = np.zeros((len(values), len(vocabulary) + 1))
 
    for row_number in range(len(values)):
        value = values[row_number]
        if value in position_of:
            column = position_of[value]
        else:
            column = other_position
        matrix[row_number, column] = 1.0
 
    column_names = []
    for category in vocabulary:
        column_names.append(category)
    column_names.append(OTHER_COLUMN_NAME)
 
    return matrix, column_names

# multi-hot encoding; needed for projects that have multiple categories ex. "Bridges, Roads"
def multi_hot(list_of_type_lists, vocabulary):
    position_of = {}
    for position in range(len(vocabulary)):
        position_of[vocabulary[position]] = position
 
    matrix = np.zeros((len(list_of_type_lists), len(vocabulary)))
 
    for row_number in range(len(list_of_type_lists)):
        for project_type in list_of_type_lists[row_number]:
            if project_type in position_of:
                matrix[row_number, position_of[project_type]] = 1.0
 
    return matrix

# TRAIN AND TEST SPLIT BASED ON THE PROJECTS "MATURITY" 

def get_start_date(project):
    return project["start_date"]

def split_by_maturity(projects, snapshot_date, maturity_days):
    cutoff_date = snapshot_date - datetime.timedelta(days=maturity_days)
 
    matured_projects = []
    recent_projects = []
    for project in projects:
        if project["start_date"] <= cutoff_date:
            matured_projects.append(project)
        else:
            recent_projects.append(project)
 
    return matured_projects, recent_projects, cutoff_date

def chronological_split(projects, train_fraction):
    ordered_projects = sorted(projects, key=get_start_date)
    total = len(ordered_projects)
 
    cut = int(total * train_fraction)
    while cut < total:
        if ordered_projects[cut]["start_date"] == ordered_projects[cut - 1]["start_date"]:
            cut = cut + 1
        else:
            break
 
    train_projects = ordered_projects[:cut]
    test_projects = ordered_projects[cut:]

    return train_projects, test_projects

# FEATURE BUILDING

def fit_parameters(train_projects):
    all_types = []
    for project in train_projects:
        for project_type in project["types"]:
            all_types.append(project_type)
 
    # Log of every known budget; the median of these fills missing budgets.
    log_budgets = []
    for project in train_projects:
        if project["budget"] is not None:
            log_budgets.append(math.log1p(project["budget"]))
 
    parameters = {
        "type_vocabulary": fit_vocabulary(all_types, 1),
        "region_vocabulary": fit_vocabulary(
            collect_values(train_projects, "region"), MIN_COUNT_REGION),
        "province_vocabulary": fit_vocabulary(
            collect_values(train_projects, "province"), MIN_COUNT_PROVINCE),
        "median_log_budget": median(log_budgets),
        "contractor_counts": count_values(
            collect_values(train_projects, "contractor_id")),
    }
    return parameters

def build_features(projects, parameters, is_train):
    log_budgets = []
    budget_missing_flags = []
    contractor_experience = []
    revoked_flags = []
 
    for project in projects:
        # ---- Budget: fill if missing, then take log(1 + budget). ------------
        # Budgets range from about 10^5 to 2x10^9; the log squeezes this into a
        # manageable range so a few huge budgets do not dominate.
        if project["budget"] is None:
            log_budgets.append(parameters["median_log_budget"])
            budget_missing_flags.append(1.0)
        else:
            log_budgets.append(math.log1p(project["budget"]))
            budget_missing_flags.append(0.0)
 
        # ---- Contractor experience. -----------------------------------------
        # How many TRAIN projects did this contractor have? Use log(1 + count)
        # to soften very large counts. A TRAIN row counts itself in the table,
        # so for TRAIN rows we subtract 1; this way train and test rows are
        # measured the same way (the number of OTHER projects seen in training).
        count = 0
        if project["contractor_id"] in parameters["contractor_counts"]:
            count = parameters["contractor_counts"][project["contractor_id"]]
        if is_train:
            if count > 0:
                count = count - 1
        contractor_experience.append(math.log1p(count))
 
        revoked_flags.append(project["revoked"])
 
    # Numeric columns in a fixed order: (name, values, is_continuous).
    numeric_columns = [
        ("log_budget", log_budgets, True),
        ("budget_missing", budget_missing_flags, False),
        ("contractor_revoked", revoked_flags, False),
        ("log_contractor_train_projects", contractor_experience, True),
    ]
 
    feature_names = []
    continuous_positions = []
    column_arrays = []
 
    for position in range(len(numeric_columns)):
        name, values, is_continuous = numeric_columns[position]
        feature_names.append(name)
        column_arrays.append(np.array(values, dtype=float))
        if is_continuous:
            continuous_positions.append(position)
 
    blocks = [np.column_stack(column_arrays)]
 
    # ---- Project type: multi-hot (a project may have several types). --------
    list_of_type_lists = collect_values(projects, "types")
    blocks.append(multi_hot(list_of_type_lists, parameters["type_vocabulary"]))
    for project_type in parameters["type_vocabulary"]:
        feature_names.append("type=" + project_type)
 
    # ---- Region and province (engineering office): one-hot. -----------------
    categorical_specs = [
        ("region_vocabulary", "region", "region="),
        ("province_vocabulary", "province", "office="),
    ]
    for vocabulary_key, field, prefix in categorical_specs:
        block, column_names = one_hot(
            collect_values(projects, field), parameters[vocabulary_key])
        blocks.append(block)
        for column_name in column_names:
            feature_names.append(prefix + column_name)
 
    matrix = np.hstack(blocks)
    return matrix, feature_names, continuous_positions

# MAIN

def preprocess(path):
    projects = read_projects(path)
 
    # ---- Keep only projects that had a fair two years to finish. ------------
    matured_projects, recent_projects, cutoff_date = split_by_maturity(
        projects, SNAPSHOT_DATE, MATURITY_DAYS)
 
    # ---- Train / test split by date (older = train, newer = test). -----------
    train_projects, test_projects = chronological_split(
        matured_projects, TRAIN_FRACTION)
 
    print("Snapshot %s | cutoff %s | matured %d | recent %d" % (
        SNAPSHOT_DATE, cutoff_date, len(matured_projects), len(recent_projects)))
    print("Train: %s to %s | Test: %s to %s" % (
        train_projects[0]["start_date"], train_projects[-1]["start_date"],
        test_projects[0]["start_date"], test_projects[-1]["start_date"]))
 
    # ---- Learn on TRAIN only, then apply the same recipe to every split. ----
    parameters = fit_parameters(train_projects)
 
    X_train, feature_names, continuous_positions = build_features(
        train_projects, parameters, is_train=True)
    X_test, _, _ = build_features(
        test_projects, parameters, is_train=False)
    X_recent, _, _ = build_features(
        recent_projects, parameters, is_train=False)
 
    # ---- Standardize continuous columns (z-score). --------------------------
    #   mean = (1/n) * sum(x)
    #   std  = sqrt( (1/n) * sum((x - mean)^2) )
    #   z    = (x - mean) / std
    # Mean and std come from TRAIN only. After this, each continuous TRAIN
    # column has mean 0 and std 1, so no column dominates distance-based
    # methods such as KNN just because its numbers are larger.
    train_continuous = X_train[:, continuous_positions]
    column_means = np.sum(train_continuous, axis=0) / train_continuous.shape[0]
    squared_differences = (train_continuous - column_means) ** 2
    column_stds = np.sqrt(np.sum(squared_differences, axis=0) / train_continuous.shape[0])
 
    # A column that never changes has std 0; use 1 so we never divide by zero.
    for position in range(len(column_stds)):
        if column_stds[position] == 0.0:
            column_stds[position] = 1.0
 
    split_matrices = [X_train, X_test, X_recent]
    for matrix in split_matrices:
        matrix[:, continuous_positions] = (
            (matrix[:, continuous_positions] - column_means) / column_stds)
 
    # ---- Labels: 0 = finished, 1 = not finished. ----------------------------
    train_labels = collect_values(train_projects, "label")
    test_labels = collect_values(test_projects, "label")
    recent_labels = collect_values(recent_projects, "label")
 
    result = {
        "X_train": X_train,
        "X_test": X_test,
        "X_recent": X_recent,
        "y_train": np.array(train_labels),
        "y_test": np.array(test_labels),
        "y_recent": np.array(recent_labels),
        "feature_names": np.array(feature_names),
    }
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "../datasets/edited_dataset.csv"

    data = preprocess(csv_path)
    with open("train_features.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(data["feature_names"]) + ["label"])
        for row, label in zip(data["X_train"], data["y_train"]):
            writer.writerow(list(row) + [label])

    for key in data:
        print("%-14s %s" % (key, data[key].shape))

    print("Unfinished projects: train %.1f%% | test %.1f%% | recent %.1f%%" % (
        100.0 * np.sum(data["y_train"]) / len(data["y_train"]),
        100.0 * np.sum(data["y_test"]) / len(data["y_test"]),
        100.0 * np.sum(data["y_recent"]) / len(data["y_recent"])))