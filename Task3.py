import pandas as pd


# Load dataset
data = pd.read_csv("GenreClassData_30s.txt", sep="\t")  # adjust if needed


# Define features and target
#I want to design a filter that can remove features that have a variance below a certain threshold
# The three features are predefined which are spectral_rolloff_mean, mfcc_1_mean, spectral_centroid_mean
# The forth one is determined by the filter
# Now we should create a function called filter_features that takes in the data and the threshold and returns the filtered

def filter_features(data, threshold):
    # Define features and target
    features = ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean"]
    target = "GenreID"

    # Extract features and target
    X = data[features]
    y = data[target]

    #Calculate variance of each feature
    variances = X.var()
    print(variances)

    # Filter features based on variance but implemented manually without importing any libraries
    selected_features = []
    for feature in X.columns:
        if X[feature].var() >= threshold:
            selected_features.append
    

    return selected_features

# Test the function
threshold = 0.1  # adjust if needed
selected_feature = filter_features(data, threshold)
print(selected_feature)




