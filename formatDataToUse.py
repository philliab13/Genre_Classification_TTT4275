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
        
        
def create_data(feature_map, features, target, partition):
    training_data = []
    training_target=[]
    testing_data = []
    testing_labels = []
    n_training = round(len(feature_map[features[0]])*partition)
    n_testing = round(len(feature_map[features[0]])*(1-partition))
    print(n_training,n_testing)

    for i in range(n_training):

        feature_vector = [feature_map[feat][i] for feat in features]

        target_value = feature_map[target][i] if target in feature_map else None

        training_data.append(feature_vector)
        training_target.append(target_value)

    for i in range(n_training,n_testing+n_training,1):
        feature_vector = [feature_map[feat][i] for feat in features]
        target_value = feature_map[target][i] if target in feature_map else None
        
        testing_data.append(feature_vector)
        testing_labels.append(target_value)

    return training_data, training_target, testing_data, testing_labels

        

