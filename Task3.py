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
from Task_1 import *

def forward_selection(base_features,candidate_features,performance_metric,data):
    performance_history=[]
    rem_feat=candidate_features
    sel_feat=base_features

    best_score=0
    improvment=True


    while improvment and rem_feat:
        improvment=False
        scores_candidates=[]

        for feature in rem_feat:


            temp_feat=sel_feat+ candidate_features[-1] 
            temp_feat_with_label=sel_feat+ candidate_features[-1] + "Track ID"+ "GenreID"
            feat_dict=get_features_dict(data, temp_feat_with_label)

            training_data, training_labels, testing_data, testing_labels=create_data(feat_dict, temp_feat, "GenreID", 0.8)
            

            #Get the feature matrix with featues and values for each

            #Do score validation
        best_candidate, score= max(scores, lambda)

        if score > best_score:
            best_score = score
            sel_feat.append(best_candidate)
            rem_feat.remove(best_candidate)
            improvement = True
            performance_history.append((list(sel_feat), best_score))
            print(f"Added feature {best_candidate}, new score: {best_score:.4f}")
        else:
            # No improvement, stop selection.
            break

    return sel_feat, performance_history



