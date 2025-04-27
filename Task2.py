from formatDataToUse import *
import matplotlib.pyplot as plt
import itertools
import numpy as np

# Task 2: Look at scatterplots or histograms of the data to see how the features for different genres overlap.
# Pop is class 1, metal 2, disco 3, classical 6.
data = get_data("GenreClassData_30s.txt")
feat_dict = get_features_dict(data, ["Track ID", "spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo", "GenreID"])

def extractFeatureInClass(feature, class_, data):
    liste = []
    for genre, value in zip(data["GenreID"], data[feature]):
        if genre == class_:
            liste.append(value)
    return liste

# Map class numbers to actual genres.
genre_names = {1: "Pop", 2: "Metal", 3: "Disco", 6: "Classical"}

# Extract data for each feature and class
tempo_class1 = extractFeatureInClass("tempo", 1, data)
tempo_class2 = extractFeatureInClass("tempo", 2, data)
tempo_class3 = extractFeatureInClass("tempo", 3, data)
tempo_class6 = extractFeatureInClass("tempo", 6, data)

spectral_rolloff_mean_class1 = extractFeatureInClass("spectral_rolloff_mean", 1, data)
spectral_rolloff_mean_class2 = extractFeatureInClass("spectral_rolloff_mean", 2, data)
spectral_rolloff_mean_class3 = extractFeatureInClass("spectral_rolloff_mean", 3, data)
spectral_rolloff_mean_class6 = extractFeatureInClass("spectral_rolloff_mean", 6, data)

mfcc_1_mean_class1 = extractFeatureInClass("mfcc_1_mean", 1, data)
mfcc_1_mean_class2 = extractFeatureInClass("mfcc_1_mean", 2, data)
mfcc_1_mean_class3 = extractFeatureInClass("mfcc_1_mean", 3, data)
mfcc_1_mean_class6 = extractFeatureInClass("mfcc_1_mean", 6, data)

spectral_centroid_mean_class1 = extractFeatureInClass("spectral_centroid_mean", 1, data)
spectral_centroid_mean_class2 = extractFeatureInClass("spectral_centroid_mean", 2, data)
spectral_centroid_mean_class3 = extractFeatureInClass("spectral_centroid_mean", 3, data)
spectral_centroid_mean_class6 = extractFeatureInClass("spectral_centroid_mean", 6, data)

def plotAllFeatures():
    # Define colors for each class
    colors = {1: "blue", 2: "red", 3: "green", 6: "black"}

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()  # Flatten for easier indexing

    # Plot Feature 1: Tempo
    axs[0].hist(tempo_class1, bins=10, alpha=0.6, label=genre_names[1], color=colors[1], density=True, histtype="step")
    axs[0].hist(tempo_class2, bins=10, alpha=0.6, label=genre_names[2], color=colors[2], density=True, histtype="step")
    axs[0].hist(tempo_class3, bins=10, alpha=0.6, label=genre_names[3], color=colors[3], density=True, histtype="step")
    axs[0].hist(tempo_class6, bins=10, alpha=0.6, label=genre_names[6], color=colors[6], density=True, histtype="step")
    axs[0].set_title("Tempo")
    axs[0].set_xlabel("Tempo")
    axs[0].set_ylabel("Density")
    axs[0].legend()

    # Plot Feature 2: Spectral Rolloff Mean
    axs[1].hist(spectral_rolloff_mean_class1, bins=10, alpha=0.6, label=genre_names[1], color=colors[1], density=True, histtype="step")
    axs[1].hist(spectral_rolloff_mean_class2, bins=10, alpha=0.6, label=genre_names[2], color=colors[2], density=True, histtype="step")
    axs[1].hist(spectral_rolloff_mean_class3, bins=10, alpha=0.6, label=genre_names[3], color=colors[3], density=True, histtype="step")
    axs[1].hist(spectral_rolloff_mean_class6, bins=10, alpha=0.6, label=genre_names[6], color=colors[6], density=True, histtype="step")
    axs[1].set_title("Spectral Rolloff Mean")
    axs[1].set_xlabel("Spectral Rolloff Mean")
    axs[1].set_ylabel("Density")
    axs[1].legend()

    # Plot Feature 3: MFCC 1 Mean
    axs[2].hist(mfcc_1_mean_class1, bins=10, alpha=0.6, label=genre_names[1], color=colors[1], density=True, histtype="step")
    axs[2].hist(mfcc_1_mean_class2, bins=10, alpha=0.6, label=genre_names[2], color=colors[2], density=True, histtype="step")
    axs[2].hist(mfcc_1_mean_class3, bins=10, alpha=0.6, label=genre_names[3], color=colors[3], density=True, histtype="step")
    axs[2].hist(mfcc_1_mean_class6, bins=10, alpha=0.6, label=genre_names[6], color=colors[6], density=True, histtype="step")
    axs[2].set_title("MFCC 1 Mean")
    axs[2].set_xlabel("MFCC 1 Mean")
    axs[2].set_ylabel("Density")
    axs[2].legend()

    # Plot Feature 4: Spectral Centroid Mean
    axs[3].hist(spectral_centroid_mean_class1, bins=10, alpha=0.6, label=genre_names[1], color=colors[1], density=True, histtype="step")
    axs[3].hist(spectral_centroid_mean_class2, bins=10, alpha=0.6, label=genre_names[2], color=colors[2], density=True, histtype="step")
    axs[3].hist(spectral_centroid_mean_class3, bins=10, alpha=0.6, label=genre_names[3], color=colors[3], density=True, histtype="step")
    axs[3].hist(spectral_centroid_mean_class6, bins=10, alpha=0.6, label=genre_names[6], color=colors[6], density=True, histtype="step")
    axs[3].set_title("Spectral Centroid Mean")
    axs[3].set_xlabel("Spectral Centroid Mean")
    axs[3].set_ylabel("Density")
    axs[3].legend()

    plt.tight_layout()  # Adjust subplot spacing
    # Save the figure with a representative name
    plt.savefig("AllFeatures.png")
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
            ax.hist(class_data[c1], bins=10, alpha=0.6, label=genre_names[c1],
                    color=colors[c1], density=True)
            ax.hist(class_data[c2], bins=10, alpha=0.6, label=genre_names[c2],
                    color=colors[c2], density=True)
            ax.set_title(f"{feature_name}: {genre_names[c1]} vs {genre_names[c2]}")
            ax.set_xlabel(feature_name)
            ax.set_ylabel("Density")
            ax.legend()
        
        # Remove any extra subplot axes if they exist
        for j in range(len(pairs), len(axs)):
            fig.delaxes(axs[j])
        
        fig.suptitle(f"Pairwise Histogram Comparisons for {feature_name}", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Save each figure with a descriptive name that removes spaces
        fig.savefig(f"FeatureComparison_{feature_name.replace(' ', '')}.png")
        plt.show()

# Call the plotting functions
plotAllFeatures()
plotEveryFeatureCompared()

# Conclusion:
# From the histograms, there is significant overlap among the features, which explains the low accuracy for the k-NN classifier.
# The pairwise feature comparisons further highlight that some classes are easier to distinguish than others.
