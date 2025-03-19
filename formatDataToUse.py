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
        
        


