#Comparing tempo and spectral_centroid_mean from task 2, one can see that there is less overlap in the spectral_centroid_mean compared to tempo
#This can probably be a good way of desciding the last feature:
# You can compute a feature’s ANOVA F-score manually by comparing the variance between classes to the variance within classes. In other words, for each feature you’d do the following:

# Compute the overall mean of the feature.
# For each class:
# Compute the class mean.
# Compute the contribution to the between-class sum of squares: multiply the number of samples in the class by the squared difference between the class mean and the overall mean.
# Compute the contribution to the within-class sum of squares: sum the squared differences between each sample in that class and the class mean.
# Calculate mean squares:
# Mean square between (MSB) = (Between-class sum of squares) / (number of classes – 1)
# Mean square within (MSW) = (Within-class sum of squares) / (total number of samples – number of classes)
# F-score: The ratio 
# 𝐹
# =
# MSB
# /
# MSW
# F=MSB/MSW.



#Should probably also look in the book if there is a way they are choosing features, and use the same method. 

#Wrapper methods: There are many different wrapper methods, one can use backaward selection, which starts with all the different features and eliminates one after another to increase the accuracy. This is computantionally heavy,  therefore one could try forward selection instead where I can start with the 3 features already selected and add 1 after another features and see which makes the accuracy the highest. 
from formatDataToUse import *
from Knn import *
from copy import deepcopy
import sklearn.metrics as metrics

def forward_selection(base_features,candidate_features,data):
    performance_history=[]
    rem_feat=candidate_features.copy()
    sel_feat=base_features.copy()

    best_score=0
    improvement=True


    while improvement and rem_feat:
        improvement=False
        scores_candidates=[]

        for feature in rem_feat:


            temp_feat=sel_feat+ [feature]
            temp_feat_with_label=sel_feat+ [feature] + ["Track ID","GenreID"]
            feat_dict=get_features_dict(data, temp_feat_with_label)

            training_data, training_labels, testing_data, testing_labels=create_data(feat_dict, temp_feat, "GenreID", 0.8)
            pred=knn_predict(training_data,training_labels, testing_data,5)
            acc=metrics.accuracy_score(testing_labels,pred)

            scores_candidates.append((feature,acc))

           
        best_candidate, score = max(scores_candidates, key=lambda x: x[1])

        if score > best_score:
            best_score = score
            sel_feat.append(best_candidate)
            rem_feat.remove(best_candidate)
            improvement = True
            performance_history.append((deepcopy(sel_feat), best_score))
            print(f"Added feature {best_candidate}, new score: {best_score:.4f}")
        else:
            # No improvement, stop selection.
            break

    return sel_feat, performance_history

data=get_data("GenreClassData_30s.txt")
allFeatures=get_all_features(data)
print(allFeatures)
feat, hist = forward_selection(["spectral_rolloff_mean","mfcc_1_mean", "spectral_centroid_mean"],allFeatures,data)
print(feat)
