# Preprocessing the NASA Global Landslide Catalog Dataset
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
import os
import sys
import numpy as np

# DEFINITION OF CONSTANTS

# date threshold for any temporal operations
SNAPSHOT_DATE = datetime.date(2018, 2, 15)

# days window
MATURITY_DAYS = 365

# percent of data used for training (0.70); remaining (0.30) becomes test set
TRAIN_FRACTION = 0.70       

# minimum frequency for categories to get their own one-hot column
MIN_COUNT_CATEGORY = 1
MIN_COUNT_TRIGGER = 1
MIN_COUNT_SIZE = 1
MIN_COUNT_SETTING = 1
MIN_COUNT_ACCURACY = 1

# columns we read from the landslide .csv dataset
DATA_COLUMNS = [
    "fatality_count",
    "landslide_category",
    "landslide_trigger",
    "landslide_size",
    "landslide_setting",
    "location_accuracy",
    "latitude",
    "longitude",
    "event_date",
    "admin_division_population",
]

# text values that really mean "no value" (blank cells, Excel errors)
MISSING_TEXT = ["", "#ref!", "#n/a", "#value!", "nan", "null", "none"]

OTHER_COLUMN_NAME = "__OTHER__"

# PARSING VALUES

# Strip whitespaces; turn blank / error strings into None
def clean(text):
    if text is None:
        return None
    text = str(text).lower().strip()
    if text in MISSING_TEXT:
        return None
    
    return text

# Parse a number; return None if missing / not a number
def to_float(text):
    if text is None:
        return None
    try:
        x = float(text)
    except (ValueError, TypeError):
        return None
    if math.isfinite(x):
        return x
    
    return None

# Convert a date written as day/month/year to a date object
def parse_date(text):
    if text is None:
        return None

    cleaned = clean(text)
    if not cleaned:
        return None

    possible_formats = [
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y",
        "%Y-%m-%d",
    ]
    for date_format in possible_formats:
        try:
            parsed = datetime.datetime.strptime(cleaned, date_format)
            return parsed.date()
        except ValueError:
            continue

    try:
        part = cleaned.split()[0]
        return datetime.datetime.strptime(part, "%m/%d/%Y").date()
    except (ValueError, IndexError):
        return None

# Extract month (numeric 1-12) from event date
def parse_month(text):
    dt = parse_date(text)
    if dt is not None:
        return float(dt.month)
    return None

# Target label: 1 if fatality_count > 0, otherwise 0
def fatality_to_label(fatality_val):
    if fatality_val is None:
        return None
    if fatality_val > 0:
        return 1
    return 0

# READING THE .CSV FILE

# map each column name to the position of its first occurrence
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
    if column_name not in positions:
        return None
    position = positions[column_name]

    if position >= len(row):
        return None
    
    return clean(row[position])

# read the .csv file and return a list of dictionaries, one per landslide event
def read_projects(path):
    if not os.path.exists(path):
        candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "landslides.csv")
        if os.path.exists(candidate):
            path = candidate

    projects = []
 
    with open(path, newline="", encoding="utf-8-sig", errors="ignore") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        positions = find_column_positions(header)
 
        for column_name in ["fatality_count"]:
            if column_name not in positions:
                raise ValueError("Column missing from the CSV: " + column_name)

        # Read the raw (still text) value of every column.
        for row in reader:
            raw = {}
            for column_name in DATA_COLUMNS:
                raw[column_name] = read_cell(row, positions, column_name)
 
            # Drop rows with blank / missing fatality count (target)
            if raw["fatality_count"] is None:
                continue

            fatality_val = to_float(raw["fatality_count"])
            if fatality_val is None:
                continue

            label = fatality_to_label(fatality_val)

            # Categorical features (blanks set to "unknown")
            category = raw["landslide_category"] or "unknown"
            trigger = raw["landslide_trigger"] or "unknown"
            size = raw["landslide_size"] or "unknown"
            setting = raw["landslide_setting"] or "unknown"
            accuracy_val = raw["location_accuracy"] or "unknown"

            # Numeric features
            latitude = to_float(raw["latitude"])
            longitude = to_float(raw["longitude"])
            month = parse_month(raw["event_date"])

            pop = to_float(raw["admin_division_population"])
            logpop = math.log1p(max(0.0, pop)) if pop is not None else None

            start_date = parse_date(raw["event_date"]) or datetime.date(2010, 1, 1)

            event = {
                "label": label,
                "landslide_category": category,
                "landslide_trigger": trigger,
                "landslide_size": size,
                "landslide_setting": setting,
                "location_accuracy": accuracy_val,
                "latitude": latitude,
                "longitude": longitude,
                "month": month,
                "logpop": logpop,
                "start_date": start_date,
            }
            projects.append(event)

    return projects

# HELPER FUNCTIONS

def median(values):
    valid = sorted([v for v in values if v is not None and math.isfinite(v)])
    if not valid:
        return 0.0
    count = len(valid)
    middle = count // 2
 
    if count % 2 == 1:
        return valid[middle]
 
    return (valid[middle - 1] + valid[middle]) / 2.0

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
    return (-count, category)

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

# TRAIN AND TEST SPLIT

def get_start_date(project):
    return project["start_date"]

def split_by_maturity(projects, snapshot_date, maturity_days):
    # In the landslide catalog, all events are terminal/recorded outcomes
    return list(projects), [], snapshot_date

def chronological_split(projects, train_fraction):
    ordered_projects = sorted(projects, key=get_start_date)
    total = len(ordered_projects)
 
    cut = int(total * train_fraction)
    while cut < total and cut > 0:
        if ordered_projects[cut]["start_date"] == ordered_projects[cut - 1]["start_date"]:
            cut = cut + 1
        else:
            break
 
    train_projects = ordered_projects[:cut]
    test_projects = ordered_projects[cut:]

    return train_projects, test_projects

# FEATURE BUILDING

def fit_parameters(train_projects):
    parameters = {
        "category_vocabulary": fit_vocabulary(
            collect_values(train_projects, "landslide_category"), MIN_COUNT_CATEGORY),
        "trigger_vocabulary": fit_vocabulary(
            collect_values(train_projects, "landslide_trigger"), MIN_COUNT_TRIGGER),
        "size_vocabulary": fit_vocabulary(
            collect_values(train_projects, "landslide_size"), MIN_COUNT_SIZE),
        "setting_vocabulary": fit_vocabulary(
            collect_values(train_projects, "landslide_setting"), MIN_COUNT_SETTING),
        "accuracy_vocabulary": fit_vocabulary(
            collect_values(train_projects, "location_accuracy"), MIN_COUNT_ACCURACY),
    }

    # Numerics: compute medians for imputation, and mean/std for z-score standardization
    for col in ["latitude", "longitude", "month", "logpop"]:
        vals = [p[col] for p in train_projects if p[col] is not None]
        med = median(vals)
        parameters[col + "_median"] = med

        imputed = [p[col] if p[col] is not None else med for p in train_projects]
        mean = float(np.mean(imputed))
        std = float(np.std(imputed))
        parameters[col + "_mean"] = mean
        parameters[col + "_std"] = std if std > 1e-6 else 1.0

    return parameters

def build_features(projects, parameters, is_train=True):
    # ---- 4 Numeric Columns (Imputed with median, Standardized Mean=0, SD=1) ---
    continuous_columns = ["latitude", "longitude", "month", "logpop"]
    
    feature_names = []
    continuous_positions = []
    column_arrays = []

    for pos, col in enumerate(continuous_columns):
        med = parameters[col + "_median"]
        mean = parameters[col + "_mean"]
        std = parameters[col + "_std"]

        col_vals = []
        for p in projects:
            val = p[col] if p[col] is not None else med
            col_vals.append((val - mean) / std)

        feature_names.append(col)
        column_arrays.append(np.array(col_vals, dtype=float))
        continuous_positions.append(pos)

    blocks = [np.column_stack(column_arrays)]

    # ---- 5 Categorical Columns (One-hot encoded) ----------------------------
    categorical_specs = [
        ("category_vocabulary", "landslide_category", "category="),
        ("trigger_vocabulary", "landslide_trigger", "trigger="),
        ("size_vocabulary", "landslide_size", "size="),
        ("setting_vocabulary", "landslide_setting", "setting="),
        ("accuracy_vocabulary", "location_accuracy", "accuracy="),
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
 
    matured_projects, recent_projects, cutoff_date = split_by_maturity(
        projects, SNAPSHOT_DATE, MATURITY_DAYS)
 
    train_projects, test_projects = chronological_split(
        matured_projects, TRAIN_FRACTION)
 
    print("Total events: %d | Train: %d | Test: %d" % (
        len(matured_projects), len(train_projects), len(test_projects)))
 
    # Learn on TRAIN only, then apply parameters
    parameters = fit_parameters(train_projects)
 
    X_train, feature_names, continuous_positions = build_features(
        train_projects, parameters, is_train=True)
    X_test, _, _ = build_features(
        test_projects, parameters, is_train=False)
 
    train_labels = collect_values(train_projects, "label")
    test_labels = collect_values(test_projects, "label")
 
    result = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": np.array(train_labels),
        "y_test": np.array(test_labels),
        "feature_names": np.array(feature_names),
    }
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "../datasets/landslides.csv"

    data = preprocess(csv_path)
    for key in data:
        print("%-14s %s" % (key, data[key].shape))

    print("Fatal events: train %.1f%% | test %.1f%%" % (
        100.0 * np.sum(data["y_train"]) / len(data["y_train"]),
        100.0 * np.sum(data["y_test"]) / len(data["y_test"])))