import numpy as np
from collections import Counter

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