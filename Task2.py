from formatDataToUse import *
import matplotlib.pyplot as plt
import itertools

#Task 2, look at scatterplot or histograms of the data, that could explain why the classifier has low accuracy. The task asks you to look at each of the different features in the different genres and see how much they overlap. So one plot is the tempo of each genre, and you see how much they overlap for each of them, the more they overlap, the harder it is for the classifier to classify them.

#Pop is class 1, metal 2, disco 3, calssical 6,
data=get_data("GenreClassData_30s.txt")
feat_dict=get_features_dict(data, ["Track ID", "spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean","tempo", "GenreID"])

def extractFeatureInClass(feature, class_, data):
    liste=[]
    for genre, temp in zip(data["GenreID"], data[feature]):
        if genre==class_:
            liste.append(temp)
        
    return liste
    

# Extract data for each feature and class
# Feature 1: Tempo
tempo_class1 = extractFeatureInClass("tempo", 1, data)
tempo_class2 = extractFeatureInClass("tempo", 2, data)
tempo_class3 = extractFeatureInClass("tempo", 3, data)
tempo_class6 = extractFeatureInClass("tempo", 6, data)

# Feature 2: Spectral Rolloff Mean
spectral_rolloff_mean_class1 = extractFeatureInClass("spectral_rolloff_mean", 1, data)
spectral_rolloff_mean_class2 = extractFeatureInClass("spectral_rolloff_mean", 2, data)
spectral_rolloff_mean_class3 = extractFeatureInClass("spectral_rolloff_mean", 3, data)
spectral_rolloff_mean_class6 = extractFeatureInClass("spectral_rolloff_mean", 6, data)

# Feature 3: MFCC 1 Mean
mfcc_1_mean_class1 = extractFeatureInClass("mfcc_1_mean", 1, data)
mfcc_1_mean_class2 = extractFeatureInClass("mfcc_1_mean", 2, data)
mfcc_1_mean_class3 = extractFeatureInClass("mfcc_1_mean", 3, data)
mfcc_1_mean_class6 = extractFeatureInClass("mfcc_1_mean", 6, data)

# Feature 4: Spectral Centroid Mean
spectral_centroid_mean_class1 = extractFeatureInClass("spectral_centroid_mean", 1, data)
spectral_centroid_mean_class2 = extractFeatureInClass("spectral_centroid_mean", 2, data)
spectral_centroid_mean_class3 = extractFeatureInClass("spectral_centroid_mean", 3, data)
spectral_centroid_mean_class6 = extractFeatureInClass("spectral_centroid_mean", 6, data)

def plotAllFeatures():
    # Define colors for each class
    colors = {1: "blue", 2: "red", 3: "green", 6: "black"}

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()  # Flatten to simplify indexing

    # Plot Feature 1: Tempo
    axs[0].hist(tempo_class1, bins=10, alpha=0.6, label="Class 1", color=colors[1], density=True, histtype="step")
    axs[0].hist(tempo_class2, bins=10, alpha=0.6, label="Class 2", color=colors[2], density=True, histtype="step")
    axs[0].hist(tempo_class3, bins=10, alpha=0.6, label="Class 3", color=colors[3], density=True, histtype="step")
    axs[0].hist(tempo_class6, bins=10, alpha=0.6, label="Class 6", color=colors[6], density=True, histtype="step")
    axs[0].set_title("Tempo")
    axs[0].set_xlabel("Tempo")
    axs[0].set_ylabel("Density")
    axs[0].legend()

    # Plot Feature 2: Spectral Rolloff Mean
    axs[1].hist(spectral_rolloff_mean_class1, bins=10, alpha=0.6, label="Class 1", color=colors[1], density=True, histtype="step")
    axs[1].hist(spectral_rolloff_mean_class2, bins=10, alpha=0.6, label="Class 2", color=colors[2], density=True, histtype="step")
    axs[1].hist(spectral_rolloff_mean_class3, bins=10, alpha=0.6, label="Class 3", color=colors[3], density=True, histtype="step")
    axs[1].hist(spectral_rolloff_mean_class6, bins=10, alpha=0.6, label="Class 6", color=colors[6], density=True, histtype="step")
    axs[1].set_title("Spectral Rolloff Mean")
    axs[1].set_xlabel("Spectral Rolloff Mean")
    axs[1].set_ylabel("Density")
    axs[1].legend()

    # Plot Feature 3: MFCC 1 Mean
    axs[2].hist(mfcc_1_mean_class1, bins=10, alpha=0.6, label="Class 1", color=colors[1], density=True, histtype="step")
    axs[2].hist(mfcc_1_mean_class2, bins=10, alpha=0.6, label="Class 2", color=colors[2], density=True, histtype="step")
    axs[2].hist(mfcc_1_mean_class3, bins=10, alpha=0.6, label="Class 3", color=colors[3], density=True, histtype="step")
    axs[2].hist(mfcc_1_mean_class6, bins=10, alpha=0.6, label="Class 6", color=colors[6], density=True, histtype="step")
    axs[2].set_title("MFCC 1 Mean")
    axs[2].set_xlabel("MFCC 1 Mean")
    axs[2].set_ylabel("Density")
    axs[2].legend()

    # Plot Feature 4: Spectral Centroid Mean
    axs[3].hist(spectral_centroid_mean_class1, bins=10, alpha=0.6, label="Class 1", color=colors[1], density=True, histtype="step")
    axs[3].hist(spectral_centroid_mean_class2, bins=10, alpha=0.6, label="Class 2", color=colors[2], density=True, histtype="step")
    axs[3].hist(spectral_centroid_mean_class3, bins=10, alpha=0.6, label="Class 3", color=colors[3], density=True, histtype="step")
    axs[3].hist(spectral_centroid_mean_class6, bins=10, alpha=0.6, label="Class 6", color=colors[6], density=True, histtype="step")
    axs[3].set_title("Spectral Centroid Mean")
    axs[3].set_xlabel("Spectral Centroid Mean")
    axs[3].set_ylabel("Density")
    axs[3].legend()

    plt.tight_layout()  # Adjust subplot spacing for clarity
    plt.show()

def plotEveryFeatureCompared():
    # Organize the data in a dictionary for easier looping
    features = {
        "Tempo": {
            1: tempo_class1,
            2: tempo_class2,
            3: tempo_class3,
            6: tempo_class6,
        },
        "Spectral Rolloff Mean": {
            1: spectral_rolloff_mean_class1,
            2: spectral_rolloff_mean_class2,
            3: spectral_rolloff_mean_class3,
            6: spectral_rolloff_mean_class6,
        },
        "MFCC 1 Mean": {
            1: mfcc_1_mean_class1,
            2: mfcc_1_mean_class2,
            3: mfcc_1_mean_class3,
            6: mfcc_1_mean_class6,
        },
        "Spectral Centroid Mean": {
            1: spectral_centroid_mean_class1,
            2: spectral_centroid_mean_class2,
            3: spectral_centroid_mean_class3,
            6: spectral_centroid_mean_class6,
        },
    }

    # Define the class labels and associated colors
    classes = [1, 2, 3, 6]
    colors = {1: "blue", 2: "red", 3: "green", 6: "black"}

    # Loop over each feature and plot pairwise histograms
    for feature_name, class_data in features.items():
        # Get all unique pair combinations of classes
        pairs = list(itertools.combinations(classes, 2))
        
        # Create a 2x3 grid for the 6 pairwise comparisons
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        axs = axs.flatten()  # Flatten for easy indexing

        for i, (c1, c2) in enumerate(pairs):
            ax = axs[i]
            # Plot histograms for the two classes with filled bins
            ax.hist(class_data[c1], bins=10, alpha=0.6, label=f"Class {c1}",
                    color=colors[c1], density=True)
            ax.hist(class_data[c2], bins=10, alpha=0.6, label=f"Class {c2}",
                    color=colors[c2], density=True)
            ax.set_title(f"{feature_name}: Class {c1} vs Class {c2}")
            ax.set_xlabel(feature_name)
            ax.set_ylabel("Density")
            ax.legend()
        
        # Remove any extra subplot axes if they exist
        for j in range(len(pairs), len(axs)):
            fig.delaxes(axs[j])
        
        fig.suptitle(f"Pairwise Histogram Comparisons for {feature_name}", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


plotAllFeatures()
plotEveryFeatureCompared()
#Conclusion, from the histograms, one could see that there is a lot of overlap in all of the features. This explains the low accuracy for the k-NN classifier. When plotting every feature compared to eachother you can see than some classes are easier to distingish between other than others. 