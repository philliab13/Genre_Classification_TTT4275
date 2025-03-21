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

#Recall and presicion are worth looking into in task 1, to determine how well the classifier is doing
prediction = knn_predict(training_data, training_labels, testing_data, 5)
acc=metrics.accuracy_score(testing_labels, prediction)
print(f"Accuracy testing points: {acc}")

prediction = knn_predict(training_data, training_labels, training_data, 5)
acc=metrics.accuracy_score(training_labels, prediction)
print(f"Accuracy training points: {acc}")

#Task 2, look at scatterplot or histograms of the data, that could explain why the classifier has low accuracy. The task asks you to look at each of the different features in the different genres and see how much they overlap. So one plot is the tempo of each genre, and you see how much they overlap for each of them, the more they overlap, the harder it is for the classifier to classify them.

#Task 3, find out which features are most important for the classifier, that could be done by removing one feature at a time and see how much the accuracy drops.

#Task 4, this is the largest task, this could be done using a lot of different classifiers. Can use whichever calssifier I wwant, I just have to make sure that I explain why I chose that classifier, why it is good for this task, and dont just use sklearn and build a classifier, I have to build it from scratch.


