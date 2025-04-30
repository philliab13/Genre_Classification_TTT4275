import pandas as pd
import os
import numpy as np


def get_data(file):
    file_path = os.path.join(os.getcwd(), "Classification_music", file)

    data = pd.read_csv(file_path, sep="\t")

    return data


def get_all_features(data):
    features = []
    for feat in data.columns:
        if feat != "Track ID" and feat != "File" and feat != "GenreID" and feat != "Genre" and feat != "Type":
            features.append(feat)
    return features


def one_hot_encode(labels, num_classes):
    # Create a 2D array of shape (num_classes, number_of_examples)
    one_hot = np.zeros((num_classes, len(labels)))
    for idx, label in enumerate(labels):
        one_hot[label, idx] = 1
    return one_hot


def get_features_dict(data, list_feat):
    feature_map = {}
    for feat in list_feat:
        if feat in data.columns:
            feature_map[feat] = data[feat].to_numpy()
        else:
            print(f"Warning: {feat} not found in the data columns.")

    return feature_map

# Remember you can use the last feature in data to differentiate betweentraining and test.


def create_data(feature_map, features, target, partition):
    training_data = []
    training_target = []
    testing_data = []
    testing_labels = []
    n_training = round(len(feature_map[features[0]])*partition)
    n_testing = round(len(feature_map[features[0]])*(1-partition))

    for i in range(n_training):

        feature_vector = [feature_map[feat][i] for feat in features]

        target_value = feature_map[target][i] if target in feature_map else None

        training_data.append(feature_vector)
        training_target.append(target_value)

    for i in range(n_training, n_testing+n_training, 1):
        feature_vector = [feature_map[feat][i] for feat in features]
        target_value = feature_map[target][i] if target in feature_map else None

        testing_data.append(feature_vector)
        testing_labels.append(target_value)

    return training_data, training_target, testing_data, testing_labels


def normalize_data_min_max(data):
    minimum = np.min(data, axis=0)
    maximum = np.max(data, axis=0)
    normalized_data = (data-minimum)/(maximum-minimum)
    return normalized_data


def normalize_z_score(data):
    data = np.array(data)  #
    n_samples, n_features = data.shape

    mean_each_feat = np.mean(data, axis=0)
    variance_each_feat = np.var(data, axis=0, ddof=1)
    norm_data = (data-mean_each_feat)/np.sqrt(variance_each_feat)

    return norm_data


def manual_classification_report(y_true, y_pred, target_names=None):
    classes = sorted(set(y_true) | set(y_pred))

    report_lines = []

    for cls in classes:
        tp = sum((yt == cls and yp == cls) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != cls and yp == cls) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == cls and yp != cls) for yt, yp in zip(y_true, y_pred))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0.0

        support = sum(1 for yt in y_true if yt == cls)

        # If target_names is provided, map the class to its name.
        name = target_names[classes.index(cls)] if target_names else str(cls)
        report_lines.append((name, precision, recall, f1_score, support))

    # Overall accuracy: total correct / total samples.
    accuracy = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)

    header = f"{'':>10} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
    report = header + "\n\n"
    for name, prec, rec, f1, support in report_lines:
        report += f"{name:>10} {prec:10.3f} {rec:10.3f} {f1:10.3f} {support:10}\n"

    report += "\n"
    report += f"accuracy     {accuracy:10.3f} {len(y_true):10}\n"

    # Optionally, compute macro and weighted averages.
    macro_precision = sum(x[1] for x in report_lines) / len(report_lines)
    macro_recall = sum(x[2] for x in report_lines) / len(report_lines)
    macro_f1 = sum(x[3] for x in report_lines) / len(report_lines)

    report += f"macro avg   {macro_precision:10.3f} {macro_recall:10.3f} {macro_f1:10.3f} {len(y_true):10}\n"

    weighted_precision = sum(x[1] * x[4] for x in report_lines) / len(y_true)
    weighted_recall = sum(x[2] * x[4] for x in report_lines) / len(y_true)
    weighted_f1 = sum(x[3] * x[4] for x in report_lines) / len(y_true)

    report += f"weighted avg{weighted_precision:10.3f} {weighted_recall:10.3f} {weighted_f1:10.3f} {len(y_true):10}\n"

    return report

# Source: https://www.geeksforgeeks.org/binomial-coefficient-dp-9/


def binomialCoeff(n, k):

    # k can not be grater then k so we
    # return 0 here
    if k > n:
        return 0

    # base condition when k and n are equal
    # or k = 0
    if k == 0 or k == n:
        return 1

    # Recursive add the value
    return binomialCoeff(n - 1, k - 1) + binomialCoeff(n - 1, k)


def error_counting_approach(N_i, k_i, M):
    P_i = k_i/N_i
    P = binomialCoeff(N_i, k_i)*(P_i**k_i)*((1-P_i)**(N_i-k_i))
    return P
