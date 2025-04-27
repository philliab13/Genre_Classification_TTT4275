# Check out this article:  https://wandb.ai/mostafaibrahim17/ml-articles/reports/An-Introduction-to-Audio-Classification-with-Keras--Vmlldzo0MDQzNDUy
# Another link: https://medium.com/@waadlingaadil/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from copy import deepcopy
from plotting import *
from formatDataToUse import *
from Knn import *

np.random.seed(1307)


def forward_selection_nn(base_features, candidate_features, data, num_classes=10, trainingRounds=1000, alpha=0.01):
    """
    Selects features based on neural network performance.

    Parameters:
      base_features: list of initially selected features.
      candidate_features: list of remaining candidate features.
      data: your raw data (passed to get_features_dict and create_data).
      num_classes: number of output classes.
      trainingRounds: number of training iterations for each candidate evaluation.
      alpha: learning rate.

    Returns:
      sel_feat: list of selected features.
      performance_history: list of tuples (feature_subset, accuracy)
    """
    from copy import deepcopy
    import sklearn.metrics as metrics

    performance_history = []
    rem_feat = candidate_features.copy()
    sel_feat = base_features.copy()

    best_score = 0
    improvement = True

    while improvement and rem_feat:
        improvement = False
        scores_candidates = []

        for feature in rem_feat:
            # Build a temporary feature set by adding the candidate to the selected features.
            temp_feat = sel_feat + [feature]
            temp_feat_with_label = temp_feat + ["Track ID", "GenreID"]

            # Use your helper to extract features.
            feat_dict = get_features_dict(data, temp_feat_with_label)

            # Create training and testing data using the selected features.
            training_data, training_labels, testing_data, testing_labels = create_data(
                feat_dict, temp_feat, "GenreID", 0.8)

            # Convert data to numpy arrays and transpose so that each column is an example.
            # shape: (num_features, m_train)
            training_data = np.array(training_data).T
            # shape: (num_features, m_test)
            testing_data = np.array(testing_data).T

            # Normalize the data.
            z_norm_training_data = normalize_z_score(training_data)
            z_norm_testing_data = normalize_z_score(testing_data)

            # Convert training labels to one-hot encoding.
            Y_train = one_hot_encode(training_labels, num_classes)

            # Reinitialize the network with input dimension = len(temp_feat)
            input_dim = len(temp_feat)
            W, b = networkArchitecture(input_dim, num_classes, 2, 64)

            # Set the network parameters globally (or pass them to your train function)
            global W1, W2, W3, b1, b2, b3
            W1, W2, W3 = W[0], W[1], W[2]
            b1, b2, b3 = b[0], b[1], b[2]

            # Train the network on the current candidate subset.
            # (Assuming your train() function uses the global z_norm_training_data and Y variables.)
            # For modularity, you might modify train() to accept data and labels.
            # Here we temporarily override the globals:
            global z_norm_training_data_global, Y_global
            z_norm_training_data_global = z_norm_training_data
            Y_global = Y_train
            # Modify your train() function to use z_norm_training_data_global and Y_global
            # instead of hardcoded global variables.
            costs = train_nn(z_norm_training_data_global,
                             Y_global, trainingRounds, alpha)

            # Use the network to predict on the normalized testing data.
            predictions = predict(z_norm_testing_data)

            # Compute accuracy based on integer testing labels.
            acc = metrics.accuracy_score(testing_labels, predictions)
            scores_candidates.append((feature, acc))
            print(f"Candidate feature: {feature}, Accuracy: {acc:.4f}")

        # Choose the candidate feature with the highest accuracy.
        best_candidate, score = max(scores_candidates, key=lambda x: x[1])
        if score > best_score:
            best_score = score
            sel_feat.append(best_candidate)
            rem_feat.remove(best_candidate)
            improvement = True
            performance_history.append((deepcopy(sel_feat), best_score))
            print(
                f"Added feature {best_candidate}, new score: {best_score:.4f}")
        else:
            break  # No improvement; exit the loop.

    return sel_feat, performance_history


# 1. Network architecture
# 4 input features, 10 nodes in each of the 3 hidden layers, 10 output classes
n = [63, 64, 64, 10]


def networkArchitecture(nr_Features, nr_out, nr_hidden_layers, nodes_hidden_layer):
    weights = []
    biases = []

    if nr_hidden_layers == 0:
        W = np.random.randn(nr_out, nr_Features) * np.sqrt(2.0 / nr_Features)
        b = np.zeros((nr_out, 1))
        weights.append(W)
        biases.append(b)
    else:
        # Input to first hidden layer: shape (nodes_hidden_layer, nr_Features)
        W = np.random.randn(nodes_hidden_layer, nr_Features) * \
            np.sqrt(2.0 / nr_Features)
        b = np.zeros((nodes_hidden_layer, 1))
        weights.append(W)
        biases.append(b)

        # Hidden layers (if more than one)
        for _ in range(nr_hidden_layers - 1):
            W = np.random.randn(
                nodes_hidden_layer, nodes_hidden_layer) * np.sqrt(2.0 / nodes_hidden_layer)
            b = np.zeros((nodes_hidden_layer, 1))
            weights.append(W)
            biases.append(b)

        # Last hidden layer to output layer: shape (nr_out, nodes_hidden_layer)
        W = np.random.randn(nr_out, nodes_hidden_layer) * \
            np.sqrt(2.0 / nodes_hidden_layer)
        b = np.zeros((nr_out, 1))
        weights.append(W)
        biases.append(b)

    return weights, biases


def sigmoid(arr):
    return 1 / (1 + np.exp(-1 * arr))


def softmax(z):
    # For numerical stability, subtract the maximum value in each column
    z = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def feed_forward(data, W, b):

    # layer 1 calculations

    Z1 = W[0] @ data + b[0]
    A1 = sigmoid(Z1)

    # layer 2 calculations
    Z2 = W[1] @ A1 + b[1]
    A2 = sigmoid(Z2)

    # layer 3 calculations
    Z3 = W[2] @ A2 + b[2]
    A3 = softmax(Z3)

    y_hat = A3
    cache = {
        "A0": data,
        "A1": A1,
        "A2": A2
    }
    return y_hat, cache


def cost(y_hat, y):
    """
    y_hat: predicted probabilities, shape (num_classes, m)
    y: one-hot encoded true labels, shape (num_classes, m)
    """
    epsilon = 1e-8  # small constant to avoid log(0)
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    # Compute categorical cross-entropy loss for each example then average
    losses = -np.sum(y * np.log(y_hat), axis=0)  # sum over classes
    return np.mean(losses)


def backprop_layer_3(y_hat, Y, m, A2, W3):  # n-> nodes per layer
    A3 = y_hat

    # step 1. calculate dC/dZ3 using shorthand we derived earlier
    dC_dZ3 = (1/m) * (A3 - Y)
    # assert dC_dZ3.shape == (n[3], m)

    # step 2. calculate dC/dW3 = dC/dZ3 * dZ3/dW3
    #   we matrix multiply dC/dZ3 with (dZ3/dW3)^T
    dZ3_dW3 = A2
    # assert dZ3_dW3.shape == (n[2], m)

    dC_dW3 = dC_dZ3 @ dZ3_dW3.T
    # assert dC_dW3.shape == (n[3], n[2])

    # step 3. calculate dC/db3 = np.sum(dC/dZ3, axis=1, keepdims=True)
    dC_db3 = np.sum(dC_dZ3, axis=1, keepdims=True)
    # assert dC_db3.shape == (n[3], 1)

    # step 4. calculate propagator dC/dA2 = dC/dZ3 * dZ3/dA2
    dZ3_dA2 = W3
    dC_dA2 = W3.T @ dC_dZ3
    # assert dC_dA2.shape == (n[2], m)

    return dC_dW3, dC_db3, dC_dA2


def backprop_layer_2(propagator_dC_dA2, A1, A2, W2):

    # step 1. calculate dC/dZ2 = dC/dA2 * dA2/dZ2

    # use sigmoid derivation to arrive at this answer:
    #   sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
    #     and if a = sigmoid(z), then sigmoid'(z) = a * (1 - a)
    dA2_dZ2 = A2 * (1 - A2)
    dC_dZ2 = propagator_dC_dA2 * dA2_dZ2
    # assert dC_dZ2.shape == (n[2], m)

    # step 2. calculate dC/dW2 = dC/dZ2 * dZ2/dW2
    dZ2_dW2 = A1
    # assert dZ2_dW2.shape == (n[1], m)

    dC_dW2 = dC_dZ2 @ dZ2_dW2.T
    # assert dC_dW2.shape == (n[2], n[1])

    # step 3. calculate dC/db2 = np.sum(dC/dZ2, axis=1, keepdims=True)
    dC_db2 = np.sum(dC_dZ2, axis=1, keepdims=True)
    # assert dC_db2.shape == (n[2], 1)

    # step 4. calculate propagator dC/dA1 = dC/dZ2 * dZ2/dA1
    dZ2_dA1 = W2
    dC_dA1 = W2.T @ dC_dZ2
    # assert dC_dA1.shape == (n[2], m)

    return dC_dW2, dC_db2, dC_dA1


def backprop_layer_1(propagator_dC_dA1, A1, A0, W1):

    # step 1. calculate dC/dZ1 = dC/dA1 * dA1/dZ1

    # use sigmoid derivation to arrive at this answer:
    #   sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
    #     and if a = sigmoid(z), then sigmoid'(z) = a * (1 - a)
    dA1_dZ1 = A1 * (1 - A1)
    dC_dZ1 = propagator_dC_dA1 * dA1_dZ1
    # assert dC_dZ1.shape == (n[1], m)

    # step 2. calculate dC/dW1 = dC/dZ1 * dZ1/dW1
    dZ1_dW1 = A0
    # assert dZ1_dW1.shape == (n[0], m)

    dC_dW1 = dC_dZ1 @ dZ1_dW1.T
    # assert dC_dW1.shape == (n[1], n[0])

    # step 3. calculate dC/db1 = np.sum(dC/dZ1, axis=1, keepdims=True)
    dC_db1 = np.sum(dC_dZ1, axis=1, keepdims=True)
    # assert dC_db1.shape == (n[1], 1)

    return dC_dW1, dC_db1

   # A0-> data transpose, Y->labels, m->number datapoints


data = get_data("GenreClassData_30s.txt")
data5 = get_data("GenreClassData_5s.txt")
data10 = get_data("GenreClassData_10s.txt")


feat_dict = get_features_dict(data, ["Track ID", 'rmse_var', 'mfcc_5_mean', 'mfcc_6_mean', 'mfcc_1_std', 'mfcc_2_mean',
                              'chroma_stft_11_mean', 'mfcc_5_std', 'mfcc_1_mean', 'mfcc_4_mean', 'mfcc_3_std', 'chroma_stft_7_std', "GenreID"])
feat_dict5 = get_features_dict(data5, ["Track ID", 'rmse_var', 'mfcc_5_mean', 'mfcc_6_mean', 'mfcc_1_std', 'mfcc_2_mean',
                                       'chroma_stft_11_mean', 'mfcc_5_std', 'mfcc_1_mean', 'mfcc_4_mean', 'mfcc_3_std', 'chroma_stft_7_std', "GenreID"])
feat_dict10 = get_features_dict(data10, ["Track ID", 'rmse_var', 'mfcc_5_mean', 'mfcc_6_mean', 'mfcc_1_std', 'mfcc_2_mean',
                                         'chroma_stft_11_mean', 'mfcc_5_std', 'mfcc_1_mean', 'mfcc_4_mean', 'mfcc_3_std', 'chroma_stft_7_std', "GenreID"])
m = 792
training_data, training_labels, testing_data, testing_labels = create_data(
    feat_dict, ['rmse_var', 'mfcc_5_mean', 'mfcc_6_mean', 'mfcc_1_std', 'mfcc_2_mean', 'chroma_stft_11_mean', 'mfcc_5_std', 'mfcc_1_mean', 'mfcc_4_mean', 'mfcc_3_std', 'chroma_stft_7_std'], "GenreID", 0.8)
training_data5, training_labels5, testing_data5, testing_labels5 = create_data(
    feat_dict5, ['rmse_var', 'mfcc_5_mean', 'mfcc_6_mean', 'mfcc_1_std', 'mfcc_2_mean', 'chroma_stft_11_mean', 'mfcc_5_std', 'mfcc_1_mean', 'mfcc_4_mean', 'mfcc_3_std', 'chroma_stft_7_std'], "GenreID", 0.8)
training_data10, training_labels10, testing_data10, testing_labels10 = create_data(
    feat_dict10, ['rmse_var', 'mfcc_5_mean', 'mfcc_6_mean', 'mfcc_1_std', 'mfcc_2_mean', 'chroma_stft_11_mean', 'mfcc_5_std', 'mfcc_1_mean', 'mfcc_4_mean', 'mfcc_3_std', 'chroma_stft_7_std'], "GenreID", 0.8)
all_training_data = np.concatenate(
    (training_data, training_data5, training_data10), axis=0)
all_training_labels = np.concatenate(
    (training_labels, training_labels5, training_labels10), axis=0)
all_testing_data = np.concatenate(
    (testing_data, testing_data5, testing_data10), axis=0)
all_testing_labels = np.concatenate(
    (testing_labels, testing_labels5, testing_labels10), axis=0)

all_training_data = np.array(all_training_data).T
all_testing_data = np.array(all_testing_data).T

training_data = np.array(training_data).T
testing_data = np.array(testing_data).T

num_classes = 10  # because you have 10 output classes
Y = one_hot_encode(all_training_labels, num_classes)

W, b = networkArchitecture(11, num_classes, 2, 64)

z_norm_training_data = normalize_z_score(all_training_data)
z_norm_testing_data = normalize_z_score(all_testing_data)
print(np.shape(W[0]))
print(np.shape(all_training_data))
print(np.shape(b[0]))

W1 = W[0]
W2 = W[1]
W3 = W[2]
b1 = b[0]
b2 = b[1]
b3 = b[2]


def train_nn(training_data, Y, trainingRounds=7000, alpha=0.1):
    global W1, W2, W3, b1, b2, b3

    costs = []
    for e in range(trainingRounds):
        y_hat, cache = feed_forward(training_data, [W1, W2, W3], [b1, b2, b3])
        error = cost(y_hat, Y)
        costs.append(error)

        dC_dW3, dC_db3, dC_dA2 = backprop_layer_3(
            y_hat, Y, training_data.shape[1], A2=cache["A2"], W3=W3)
        dC_dW2, dC_db2, dC_dA1 = backprop_layer_2(
            dC_dA2, A1=cache["A1"], A2=cache["A2"], W2=W2)
        dC_dW1, dC_db1 = backprop_layer_1(
            dC_dA1, A1=cache["A1"], A0=cache["A0"], W1=W1)

        W3 = W3 - (alpha * dC_dW3)
        W2 = W2 - (alpha * dC_dW2)
        W1 = W1 - (alpha * dC_dW1)
        b3 = b3 - (alpha * dC_db3)
        b2 = b2 - (alpha * dC_db2)
        b1 = b1 - (alpha * dC_db1)

        if e % 100 == 0:
            print(f"Epoch {e}, Cost: {error:.4f}")

    return costs


def predict(data):
    # data should be normalized and have shape (features, number_of_examples)
    y_hat, _ = feed_forward(data, [W1, W2, W3], [b1, b2, b3])
    # Since y_hat is the output of a softmax, each column is a probability distribution.
    # The predicted class is the index with the maximum probability.
    predictions = np.argmax(y_hat, axis=0)
    return predictions


# af.remove("rmse_var")

# feat, hist = forward_selection_nn(
#     ["rmse_var"], af, data, num_classes=10, trainingRounds=1000, alpha=0.1)
# # Chosen features: ['rmse_var', 'mfcc_5_mean', 'mfcc_6_mean', 'mfcc_1_std', 'mfcc_2_mean', 'chroma_stft_11_mean', 'mfcc_5_std', 'mfcc_1_mean', 'mfcc_4_mean', 'mfcc_3_std', 'chroma_stft_7_std']
# print("Selected features: ", feat)
costs = train_nn(z_norm_training_data, Y, trainingRounds=7000, alpha=0.09)
predictions = predict(z_norm_testing_data)
cm = metrics.confusion_matrix(all_testing_labels, predictions)
print(metrics.accuracy_score(all_testing_labels, predictions))
plot_cm(cm, all_testing_labels, "cm_task4")
plt.plot(costs)
plt.show()
