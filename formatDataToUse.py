import pandas as pd
import os 
import numpy as np



def get_data(file):
    file_path = os.path.join(os.getcwd(), "Classification_music", file)


    data = pd.read_csv(file_path, sep="\t")
        

    return data


def get_features_dict(data, list_feat):
    feature_map = {}
    for feat in list_feat:
        if feat in data.columns:
            feature_map[feat] = data[feat].to_numpy()
        else:
            print(f"Warning: {feat} not found in the data columns.")
    
        
    return feature_map
        
        
def create_training_data(feature_map, features, target):
    training_data = []
    training_target=[]
    n_samples = len(feature_map[features[0]])-200

    for i in range(n_samples):
        # Create a feature vector for sample i using the provided features
        feature_vector = [feature_map[feat][i] for feat in features]
        # Get the corresponding target value for sample i
        target_value = feature_map[target][i] if target in feature_map else None
        # Append a tuple (feature_vector, target_value) to training_data
        training_data.append(feature_vector)
        training_target.append(target_value)

    return training_data, training_target

        

