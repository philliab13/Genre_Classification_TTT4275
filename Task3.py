import pandas as pd
import numpy as np
from formatDataToUse import *
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

# This feature selection makes use of embedded feature selection method combining the pros of filter and wrapper methods.
# The method is based on the Gini impurity measure and the decision tree algorithm.


# Load and Prepare Data
data = get_data("GenreClassData_30s.txt")
train_data = data[data["Type"] == "Train"]

# Target and feature preparation
target_column = "GenreID"
exclude_columns = ["Track ID", "File", "Genre", "Type", target_column]
X = train_data.drop(columns=exclude_columns)
y = train_data[target_column]

# Predefined features
predefined = ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean"]

# Gini Impurity => which is a measure of how impure a node is in a decision tree, i.e., how mixed the classes are.
def gini_impurity(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)

# Split Dataset
def split_dataset(X, y, feature, threshold):
    left_mask = X[feature] <= threshold
    right_mask = X[feature] > threshold
    return (X[left_mask], y[left_mask]), (X[right_mask], y[right_mask])

# Best Split for One Feature 
def best_split_for_feature(X, y, feature):
    values = X[feature].sort_values().unique()
    best_gain = 0
    best_threshold = None
    base_impurity = gini_impurity(y)

    for threshold in values:
        (X_left, y_left), (X_right, y_right) = split_dataset(X, y, feature, threshold)
        if len(y_left) == 0 or len(y_right) == 0:
            continue

        left_impurity = gini_impurity(y_left)
        right_impurity = gini_impurity(y_right)
        weighted_impurity = (len(y_left) * left_impurity + len(y_right) * right_impurity) / len(y)

        gain = base_impurity - weighted_impurity
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_gain, best_threshold

# Compute Feature Importance
def compute_feature_importance(X, y):
    feature_importance = {}
    for feature in X.columns:
        gain, _ = best_split_for_feature(X, y, feature)
        feature_importance[feature] = gain
    return feature_importance

# Select Top Feature Not in Predefined List
def embedded_feature_selection(X, y, exclude_features, top_k=1):
    available_features = [f for f in X.columns if f not in exclude_features]
    importance = compute_feature_importance(X[available_features], y)
    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    return [f for f, _ in ranked[:top_k]]

#Run Selection
best_4th_feature = embedded_feature_selection(X, y, exclude_features=predefined, top_k=1)
selected_features = predefined + best_4th_feature

print("Selected 4 features for your model:")
for i, f in enumerate(selected_features, 1):
    print(f"{i}. {f}")

# Prepare Data for Classification
feat_dict = get_features_dict(data, selected_features + ["GenreID"])
train_X, train_y, test_X, test_y = create_data(feat_dict, selected_features, "GenreID", 0.8)

#normalization of data z-score
train_X = normalize_z_score(train_X)
test_X = normalize_z_score(test_X)


# Implement k-NN (k=5)
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))

def knn_predict(train_X, train_y, test_X, k=5):
    predictions = []
    for test_point in test_X:
        distances = [(euclidean_distance(test_point, train_point), label)
                     for train_point, label in zip(train_X, train_y)]
        distances.sort(key=lambda x: x[0])
        k_labels = [label for _, label in distances[:k]]
        predictions.append(Counter(k_labels).most_common(1)[0][0])
    return predictions

# Predict and Evaluate 
predictions = knn_predict(train_X, train_y, test_X, k=5)

print("Classification Report (macro avg):")
print(classification_report(test_y, predictions, zero_division=0))


# Output:
# Selected 4 features for your model:
# 1. spectral_rolloff_mean
# 2. mfcc_1_mean
# 3. spectral_centroid_mean
# 4. rms_mean

# The reason for selecting the 4th feature is that it has the highest importance score based on the Gini impurity measure.
# This feature selection method is based on the decision tree algorithm, which is known for its feature importance computation.
# Gini impurity measure is used to evaluate the impurity of the nodes in the decision tree, and the feature with the highest gain is selected.
# The method is an embedded feature selection method, which combines the pros of filter and wrapper methods.
# 

