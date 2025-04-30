import numpy as np
import pandas as pd
from plotting import plot_cm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from formatDataToUse import *


def load_dataset(path):
    df = pd.read_csv(path, sep='\t')
    df = df.drop(columns=["Track ID", "File", "Genre"], errors="ignore")
    return df


df_5s = load_dataset("Classification_music/GenreClassData_5s.txt")
df_10s = load_dataset("Classification_music/GenreClassData_10s.txt")
df_30s = load_dataset("Classification_music/GenreClassData_30s.txt")

df_all = pd.concat([df_5s, df_10s, df_30s], axis=0).reset_index(drop=True)
print(f"Combined dataset shape: {df_all.shape}")

# Prepare features and labels
types = df_all["Type"].values
X = df_all.drop(columns=["GenreID", "Type"], errors="ignore").values
y = df_all["GenreID"].astype(int).values
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Z-score normalization
print("Normalizing features...")
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8
X_norm = (X - X_mean) / X_std

# Deterministic train/test split
is_train = types == "Train"
is_test = types == "Test"
X_train, X_test = X_norm[is_train], X_norm[is_test]
y_train, y_test = y[is_train], y[is_test]
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")


def one_hot(y, num_classes):
    return np.eye(num_classes)[y]


num_classes = len(np.unique(y))
y_train_oh = one_hot(y_train, num_classes)
y_test_oh = one_hot(y_test, num_classes)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(z):
    exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))  # stability
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def cross_entropy(predictions, targets):
    eps = 1e-9
    predictions = np.clip(predictions, eps, 1 - eps)
    return -np.sum(targets * np.log(predictions)) / predictions.shape[0]


def accuracy(preds, labels):
    return np.mean(np.argmax(preds, axis=1) == np.argmax(labels, axis=1))


print("Initializing neural network...")
np.random.seed(42)
input_size = X_train.shape[1]
hidden_size = 10  # Single hidden layer with 10 neurons
output_size = num_classes
learning_rate = 0.01
epochs = 150
batch_size = 64


W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))


print("Training neural network...")
train_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(epochs):
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train_shuffled = X_train[indices]
    y_train_oh_shuffled = y_train_oh[indices]

    batch_losses = []

    # Mini-batch gradient descent
    for start in range(0, X_train.shape[0], batch_size):
        end = min(start + batch_size, X_train.shape[0])
        x_batch = X_train_shuffled[start:end]
        y_batch = y_train_oh_shuffled[start:end]

        # Forward pass with single hidden layer
        z1 = x_batch @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        probs = softmax(z2)

        # Calculate loss
        loss = cross_entropy(probs, y_batch)
        batch_losses.append(loss)

        # Backpropagation
        dz2 = (probs - y_batch) / batch_size
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = dz2 @ W2.T
        dz1 = da1 * relu_derivative(z1)
        dW1 = x_batch.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update parameters with fixed learning rate
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

       # Evaluate after each epoch
    train_probs = softmax(relu(X_train @ W1 + b1) @ W2 + b2)
    train_acc = accuracy(train_probs, y_train_oh)

    test_probs = softmax(relu(X_test @ W1 + b1) @ W2 + b2)
    test_acc = accuracy(test_probs, y_test_oh)

    test_accuracies.append(test_acc)
    train_accuracies.append(train_acc)

    # Average loss for this epoch
    avg_loss = np.mean(batch_losses)
    train_losses.append(avg_loss)

    print(
        f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")


final_probs = softmax(relu(X_test @ W1 + b1) @ W2 + b2)
predicted_classes = np.argmax(final_probs, axis=1)
true_classes = y_test

final_probs_train = softmax(relu(X_train @ W1 + b1) @ W2 + b2)
predicted_classes_train = np.argmax(final_probs_train, axis=1)
true_classes_train = y_train

# GENRE_MAP for output
GENRE_MAP = {
    0: "pop", 1: "metal", 2: "disco", 3: "blues", 4: "reggae",
    5: "classical", 6: "rock", 7: "hiphop", 8: "country", 9: "jazz"
}

# === Calculate and Plot Confusion Matrix using sklearn ===
cm = confusion_matrix(true_classes, predicted_classes)

# Create genre name list from GENRE_MAP
genre_names = [GENRE_MAP[i] for i in range(num_classes)]


# Get complete classification report from
print("\nDetailed Classification Report from sklearn:")
report = manual_classification_report(true_classes, predicted_classes,
                                      target_names=genre_names)
print(report)

report_train = manual_classification_report(
    true_classes_train, predicted_classes_train, target_names=genre_names)
print(report_train)

# Calculate overall metrics
accuracy = np.sum(np.diag(cm)) / np.sum(cm)


print("\nAccuracy: {:.4f}".format(accuracy))


# Plot training progress
try:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('training_progress.png')
    print("\nTraining progress plot saved as 'training_progress.png'")
except Exception as e:
    print(f"\nCouldn't create plot: {e}")

# Plot the confusion matrix
print("\nConfusion Matrix:")
plot_cm(cm, genre_names, file_name="cm_task4_train_test_split.png")
print(genre_names)
# Could be interesting to try with only these features, becasue forward selection chose these as the best features: ['rmse_var', 'mfcc_5_mean', 'mfcc_6_mean', 'mfcc_1_std', 'mfcc_2_mean', 'chroma_stft_11_mean', 'mfcc_5_std', 'mfcc_1_mean', 'mfcc_4_mean', 'mfcc_3_std', 'chroma_stft_7_std']
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_progress.png')
print("\nUpdated training progress plot saved as 'training_progress.png'")
