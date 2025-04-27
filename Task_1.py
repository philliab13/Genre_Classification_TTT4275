from formatDataToUse import *
from plotting import *
from Knn import *
import matplotlib.pyplot as plt
import numpy as np

import sklearn.metrics as metrics


data=get_data("GenreClassData_30s.txt")
feat_dict=get_features_dict(data, ["Track ID", "spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean","tempo", "GenreID"])

training_data, training_labels, testing_data, testing_labels=create_data(
    feat_dict, ["spectral_rolloff_mean","mfcc_1_mean", "spectral_centroid_mean","tempo"], "GenreID", 0.8)

norm_training_data=normalize_data_min_max(training_data)
norm_testing_data=normalize_data_min_max(testing_data)
z_norm_training_data=normalize_z_score(training_data)
z_norm_testing_data=normalize_z_score(testing_data)

#Might be worth looking into tanget distance as a measure to get more accurate results
#Performance without normalizing
prediction_testing_points = knn_predict(training_data, training_labels, testing_data, 5)

classes = np.unique(testing_labels)
#You get recall, precision, f1-score from this, this could also be done by looking at the confusion matrix and using the formulas from lecture 18
report = manual_classification_report(testing_labels, prediction_testing_points, [str(cls) for cls in classes])
print(report)


cm  =metrics.confusion_matrix(testing_labels,prediction_testing_points)

plot_cm(cm,testing_labels, "cm_no_normalizing")

#Performance with normalizing
prediction_testing_points_norm = knn_predict(norm_training_data, training_labels, norm_testing_data, 5)

#You get recall, precision, f1-score from this, this could also be done by looking at the confusion matrix and using the formulas from lecture 18
report_norm = manual_classification_report(testing_labels, prediction_testing_points_norm, [str(cls) for cls in classes])
print(report_norm)


cm_norm  =metrics.confusion_matrix(testing_labels,prediction_testing_points_norm)

plot_cm(cm_norm,testing_labels,"cm_minmax_normalizing")
#Normalizing the data did not make the classifier better, overall worse, but is made some improvements individually on some classes.  

#Performance with z-score normalizing
prediction_testing_points_norm_z = knn_predict(z_norm_training_data, training_labels, z_norm_testing_data, 5)

#You get recall, precision, f1-score from this, this could also be done by looking at the confusion matrix and using the formulas from lecture 18
report_norm_z = manual_classification_report(testing_labels, prediction_testing_points_norm_z, [str(cls) for cls in classes])
print(report_norm_z)


cm_norm_z  =metrics.confusion_matrix(testing_labels,prediction_testing_points_norm_z)

plot_cm(cm_norm_z,testing_labels,"cm_z_normalizing")
#Normalizing it with the z-score seemed to help, maybe it will be worth trying to assume it is gaussian and see if that changes anything. 

