#Check out this article:  https://wandb.ai/mostafaibrahim17/ml-articles/reports/An-Introduction-to-Audio-Classification-with-Keras--Vmlldzo0MDQzNDUy
#Another link: https://medium.com/@waadlingaadil/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc

import pandas as pd
import numpy as np
from pathlib import Path
import sklearn.metrics as metrics

# === Load all datasets ===
def load_dataset(path):
    df = pd.read_csv(path, sep='\t')
    df = df.drop(columns=["Track ID", "File", "Genre", "Type"], errors="ignore")
    return df

# Load all 3 datasets
df_5s = load_dataset("Classification_music/GenreClassData_5s.txt")
df_10s = load_dataset("Classification_music/GenreClassData_10s.txt")
df_30s = load_dataset("Classification_music/GenreClassData_30s.txt")

# Combine them
df_all = pd.concat([df_5s, df_10s, df_30s], axis=0).reset_index(drop=True)

# === Prepare features and labels ===
X = df_all.drop(columns=["GenreID"]).values
y = df_all["GenreID"].astype(int).values

# === Z-score normalization ===
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8  # avoid division by zero
X_norm = (X - X_mean) / X_std

# === Train/Test split ===
split_idx = int(0.8 * len(X_norm))
X_train, X_test = X_norm[:split_idx], X_norm[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
print(len(X_train))

# === One-hot encode labels ===
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

num_classes = len(np.unique(y))
y_train_oh = one_hot(y_train, num_classes)
y_test_oh = one_hot(y_test, num_classes)

# === Neural Network Functions ===
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

# === Initialize Network ===
np.random.seed(42)
input_size = X_train.shape[1]
hidden_size = 64
output_size = num_classes
learning_rate = 0.01
epochs = 50
batch_size = 64

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# === Training Loop ===
train_losses = []
test_accuracies = []

for epoch in range(epochs):
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train_shuffled = X_train[indices]
    y_train_oh_shuffled = y_train_oh[indices]

    for start in range(0, X_train.shape[0], batch_size):
        end = start + batch_size
        x_batch = X_train_shuffled[start:end]
        y_batch = y_train_oh_shuffled[start:end]

        # Forward pass
        z1 = x_batch @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        probs = softmax(z2)

        # Backpropagation
        loss = cross_entropy(probs, y_batch)
        dz2 = (probs - y_batch) / batch_size
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = dz2 @ W2.T
        dz1 = da1 * relu_derivative(z1)
        dW1 = x_batch.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update parameters
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    # Evaluate after each epoch
    test_probs = softmax(relu(X_test @ W1 + b1) @ W2 + b2)
    test_acc = accuracy(test_probs, y_test_oh)
    test_accuracies.append(test_acc)
    train_losses.append(loss)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Test Acc: {test_acc:.4f}")

# === Final Output ===
print("\n Training complete.")
print(f"Final Loss: {train_losses[-1]:.4f}")
print(f"Final Test Accuracy: {test_accuracies[-1]*100:.2f}%")


# === Predict all test samples ===
z1_test = X_test @ W1 + b1
a1_test = relu(z1_test)
z2_test = a1_test @ W2 + b2
probs_test = softmax(z2_test)

predicted_classes = np.argmax(probs_test, axis=1)
true_classes = y_test

GENRE_MAP = {
    0: "pop", 1: "metal", 2: "disco", 3: "blues", 4: "reggae",
    5: "classical", 6: "rock", 7: "hiphop", 8: "country", 9: "jazz"
}

# Then modify the print loop:
for i in range(20):
    pred = predicted_classes[i]
    actual = true_classes[i]
    print(f"Sample {i+1}: Predicted = {GENRE_MAP[pred]} ({pred}), Actual = {GENRE_MAP[actual]} ({actual})")

cm=metrics.confusion_matrix(y_test,predicted_classes)
print(cm)