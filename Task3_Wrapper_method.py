
from formatDataToUse import *
from Knn import *
from plotting import *
import sklearn.metrics as metrics
import time

start = time.time()


def forward_selection_one_feat(base_features, candidate_features, data):
    score = []

    for feat in candidate_features:
        temp_feat = base_features + [feat]
        temp_feat_with_label = base_features + [feat] + ["Track ID", "GenreID"]
        feat_dict = get_features_dict(data, temp_feat_with_label)

        training_data, training_labels, testing_data, testing_labels = create_data(
            feat_dict, temp_feat, "GenreID", 0.8)
        z_norm_training_data = normalize_z_score(training_data)
        z_norm_testing_data = normalize_z_score(testing_data)
        pred = knn_predict(z_norm_training_data,
                           training_labels, z_norm_testing_data, 5)
        acc = metrics.accuracy_score(testing_labels, pred)

        score.append((feat, acc))
    best_candidate, score_can = max(score, key=lambda x: x[1])
    return best_candidate, score_can, score


GENRE_MAP = {
    0: "pop", 1: "metal", 2: "disco", 3: "blues", 4: "reggae",
    5: "classical", 6: "rock", 7: "hiphop", 8: "country", 9: "jazz"
}

# === Calculate and Plot Confusion Matrix using sklearn ===

# Create genre name list from GENRE_MAP
genre_names = [GENRE_MAP[i] for i in range(10)]


data = get_data("GenreClassData_30s.txt")
allFeatures = get_all_features(data)
print(allFeatures)
feat, score, liste = forward_selection_one_feat(
    ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean"], allFeatures, data)
print("Best feature: ", feat)
print("Score: ", score)
print("Complete list: ", liste)
end = time.time()

feat_dict = get_features_dict(data, ["Track ID", "spectral_rolloff_mean",
                              "mfcc_1_mean", "spectral_centroid_mean", "rmse_var", "GenreID"])

training_data, training_labels, testing_data, testing_labels = create_data(
    feat_dict, ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "rmse_var"], "GenreID", 0.8)

z_norm_training_data = normalize_z_score(training_data)
z_norm_testing_data = normalize_z_score(testing_data)

prediction_testing_points = knn_predict(
    training_data, training_labels, testing_data, 5)

classes = np.unique(testing_labels)
# You get recall, precision, f1-score from this, this could also be done by looking at the confusion matrix and using the formulas from lecture 18
report = manual_classification_report(
    testing_labels, prediction_testing_points, [str(cls) for cls in classes])
print(report)

cm = metrics.confusion_matrix(testing_labels, prediction_testing_points)

plot_cm(cm, genre_names, "task3_no_norm")

# Performance with z-score normalizing
prediction_testing_points_norm_z = knn_predict(
    z_norm_training_data, training_labels, z_norm_testing_data, 5)

# You get recall, precision, f1-score from this, this could also be done by looking at the confusion matrix and using the formulas from lecture 18
report_norm_z = manual_classification_report(
    testing_labels, prediction_testing_points_norm_z, [str(cls) for cls in classes])
print(report_norm_z)


cm_norm_z = metrics.confusion_matrix(
    testing_labels, prediction_testing_points_norm_z)

plot_cm(cm_norm_z, genre_names, "task3_norm_z")
print("Execution time: ", (end-start))
# Normalizing it with the z-score seemed to help, maybe it will be worth trying to assume it is gaussian and see if that changes anything.

# With the simplified version of forward selection it chose rmse_var as the feature that gave it the best overall accuracy, one can see that there are some genres that are very hard to classify. Look at the confusion matrix.
