from formatDataToUse import *

data=get_data("GenreClassData_30s.txt")
feat_dict=get_features_dict(data, ["Track ID", "spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean","tempo"])
print(len(feat_dict))

def k_NN_classifier():
    print("help")

