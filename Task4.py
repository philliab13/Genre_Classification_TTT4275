import numpy as np
import pandas as pd
import numpy as np
from pathlib import Path

# === Load all datasets ===
def load_dataset(path):
    df = pd.read_csv(path, sep='\t')
    # Drop Track ID, File, Genre, Type as in your original code
    df = df.drop(columns=["Track ID", "File", "Genre", "Type"], errors="ignore")
    return df

# Load all 3 datasets
print("Loading datasets...")
df_5s = load_dataset("Classification_music/GenreClassData_5s.txt")
df_10s = load_dataset("Classification_music/GenreClassData_10s.txt")
df_30s = load_dataset("Classification_music/GenreClassData_30s.txt")

# Combine them directly as in original code
df_all = pd.concat([df_5s, df_10s, df_30s], axis=0).reset_index(drop=True)
print(f"Combined dataset shape: {df_all.shape}")

# === Prepare features and labels ===
X = df_all.drop(columns=["GenreID"]).values
y = df_all["GenreID"].astype(int).values
print(f"X shape: {X.shape}, y shape: {y.shape}")

# === Z-score normalization ===
print("Normalizing features...")
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8  # avoid division by zero
X_norm = (X - X_mean) / X_std

# === Random Train/Test split (80/20) as in original code ===
np.random.seed(42)  # For reproducibility
indices = np.arange(len(X_norm))
np.random.shuffle(indices)
split_idx = int(0.8 * len(X_norm))
X_train, X_test = X_norm[:split_idx], X_norm[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

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

# === Initialize Network with single hidden layer as in original code ===
print("Initializing neural network...")
np.random.seed(42)
input_size = X_train.shape[1]
hidden_size = 64  # Single hidden layer with 64 neurons, exactly as original
output_size = num_classes
learning_rate = 0.01  # Fixed learning rate as in original
epochs = 150  # 100 epochs as in original
batch_size = 64  # 64 batch size as in original

# Use the same initialization as original code
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# === Training Loop without early stopping (as in original) ===
print("Training neural network...")
train_losses = []
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
    test_probs = softmax(relu(X_test @ W1 + b1) @ W2 + b2)
    test_acc = accuracy(test_probs, y_test_oh)
    test_accuracies.append(test_acc)
    
    # Average loss for this epoch
    avg_loss = np.mean(batch_losses)
    train_losses.append(avg_loss)
    
    # Print progress
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Test Acc: {test_acc:.4f}")

# === Final Evaluation ===
final_probs = softmax(relu(X_test @ W1 + b1) @ W2 + b2)
predicted_classes = np.argmax(final_probs, axis=1)
true_classes = y_test

# Calculate confusion matrix
def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i]][y_pred[i]] += 1
    return cm

cm = confusion_matrix(true_classes, predicted_classes, num_classes)

# Calculate precision, recall, and F1 score
def calculate_metrics(cm):
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    
    for i in range(num_classes):
        # Precision = TP / (TP + FP)
        precision[i] = cm[i, i] / max(np.sum(cm[:, i]), 1)
        
        # Recall = TP / (TP + FN)
        recall[i] = cm[i, i] / max(np.sum(cm[i, :]), 1)
        
        # F1 = 2 * (precision * recall) / (precision + recall)
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    
    return precision, recall, f1

precision, recall, f1 = calculate_metrics(cm)

# GENRE_MAP for output
GENRE_MAP = {
    0: "pop", 1: "metal", 2: "disco", 3: "blues", 4: "reggae",
    5: "classical", 6: "rock", 7: "hiphop", 8: "country", 9: "jazz"
}

# Print results
print("\nTraining complete!")
print(f"Final Test Accuracy: {test_accuracies[-1]*100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(f"{'Genre':<12} {'precision':>10} {'recall':>10} {'f1-score':>10}")
print("-" * 45)
for i in range(num_classes):
    print(f"{GENRE_MAP[i]:<12} {precision[i]:>10.2f} {recall[i]:>10.2f} {f1[i]:>10.2f}")

# Calculate overall metrics
accuracy = np.sum(np.diag(cm)) / np.sum(cm)
macro_precision = np.mean(precision)
macro_recall = np.mean(recall)
macro_f1 = np.mean(f1)

print("\nAccuracy: {:.4f}".format(accuracy))
print("Macro Precision: {:.4f}".format(macro_precision))
print("Macro Recall: {:.4f}".format(macro_recall))
print("Macro F1: {:.4f}".format(macro_f1))

# Print sample predictions
print("\nSample Predictions:")
for i in range(min(20, len(predicted_classes))):
    pred = predicted_classes[i]
    actual = true_classes[i]
    print(f"Sample {i+1}: Predicted = {GENRE_MAP[pred]} ({pred}), Actual = {GENRE_MAP[actual]} ({actual})")

