import matplotlib.pyplot as plt
import numpy as np


def plot_cm(cm, genre_names, file_name):
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Testing Data')
    plt.colorbar()

    # Get the unique classes from the testing labels
    tick_marks = np.arange(len(genre_names))
    plt.xticks(tick_marks, genre_names, rotation=45)
    plt.yticks(tick_marks, genre_names)

    # Annotate the cells with the count values
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{file_name}.png")
    plt.show()
