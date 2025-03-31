#Check out this article:  https://wandb.ai/mostafaibrahim17/ml-articles/reports/An-Introduction-to-Audio-Classification-with-Keras--Vmlldzo0MDQzNDUy
#Another link: https://medium.com/@waadlingaadil/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc
import numpy as np

# 1. Network architecture

def networkArchitecture(nr_Features, nr_out, nr_hidden_layers, nodes_hidden_layer):
    weights = []
    biases = []
    
    # Case when there are no hidden layers: input directly connected to output
    if nr_hidden_layers == 0:
        W = np.random.randn(nr_Features, nr_out) * np.sqrt(2.0 / nr_Features)
        b = np.zeros((1, nr_out))
        weights.append(W)
        biases.append(b)
    else:
        # Input to first hidden layer
        W = np.random.randn(nr_Features, nodes_hidden_layer) * np.sqrt(2.0 / nr_Features)
        b = np.zeros((1, nodes_hidden_layer))
        weights.append(W)
        biases.append(b)
        
        # Hidden layers (if more than one)
        for _ in range(nr_hidden_layers - 1):
            W = np.random.randn(nodes_hidden_layer, nodes_hidden_layer) * np.sqrt(2.0 / nodes_hidden_layer)
            b = np.zeros((1, nodes_hidden_layer))
            weights.append(W)
            biases.append(b)
        
        # Last hidden layer to output layer
        W = np.random.randn(nodes_hidden_layer, nr_out) * np.sqrt(2.0 / nodes_hidden_layer)
        b = np.zeros((1, nr_out))
        weights.append(W)
        biases.append(b)
    
    return weights, biases

def sigmoid(arr):
  return 1 / (1 + np.exp(-1 * arr))

def feed_forward(data, W,b):

  # layer 1 calculations
  Z1 = W[0] @ data + b[0]
  A1 = sigmoid(Z1)

  # layer 2 calculations
  Z2 = W[1] @ A1 + b[1]
  A2 = sigmoid(Z2)

  # layer 3 calculations
  Z3 = W[2] @ A2 + b[2]
  A3 = sigmoid(Z3)
  
  y_hat = A3
  return y_hat

def cost(y_hat, y):
  """
  y_hat should be a n^L x m matrix
  y should be a n^L x m matrix
  """
  # 1. losses is a n^L x m
  losses = - ( (y * np.log(y_hat)) + (1 - y)*np.log(1 - y_hat) )

  m = y_hat.reshape(-1).shape[0]

  # 2. summing across axis = 1 means we sum across rows, 
  #   making this a n^L x 1 matrix
  summed_losses = (1 / m) * np.sum(losses, axis=1)

  # 3. unnecessary, but useful if working with more than one node
  #   in output layer
  return np.sum(summed_losses)