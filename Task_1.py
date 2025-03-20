from formatDataToUse import *
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import sklearn.metrics as metrics


data=get_data("GenreClassData_30s.txt")
feat_dict=get_features_dict(data, ["Track ID", "spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean","tempo", "GenreID"])

training_data, training_labels, testing_data, testing_labels=create_data(
    feat_dict, ["spectral_rolloff_mean","mfcc_1_mean", "spectral_centroid_mean","tempo"], "GenreID", 0.8)

#Might be worth looking into tanget distance as a measure to get more accurate results
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))




def knn_predict(training_data, training_labels, test_points, k):
    predictions = []
    for j in range(len(test_points)):
        distances = []
        for i in range(len(training_data)):
            dist = euclidean_distance(test_points[j], training_data[i])
            distances.append((dist, training_labels[i]))
        distances.sort(key=lambda x: x[0])
        k_nearest_labels = [label for _, label in distances[:k]]
        predictions.append(Counter(k_nearest_labels).most_common(1)[0][0])
    return predictions


prediction = knn_predict(training_data, training_labels, testing_data, 5)
acc=metrics.accuracy_score(testing_labels, prediction)
print(f"Accuracy testing points: {acc}")

prediction = knn_predict(training_data, training_labels, training_data, 5)
acc=metrics.accuracy_score(training_labels, prediction)
print(f"Accuracy training points: {acc}")


