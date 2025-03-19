from formatDataToUse import *
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


data=get_data("GenreClassData_30s.txt")
feat_dict=get_features_dict(data, ["Track ID", "spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean","tempo", "GenreID"])
training_data, training_labels=create_training_data(feat_dict, ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean","tempo"], "GenreID")

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))


def knn_predict(training_data, training_labels, test_point, k):
    distances = []
    for i in range(len(training_data)):
        dist = euclidean_distance(test_point, training_data[i])
        distances.append((dist, training_labels[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    return Counter(k_nearest_labels).most_common(1)[0][0]

test_point=training_data[-1]
prediction = knn_predict(training_data, training_labels, test_point, 5)
print(prediction)


