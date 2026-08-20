import numpy as np


# ReLU activation function
# Negative values become 0, while positive values stay the same.
def relu(x):
    return np.maximum(0, x)


# Sigmoid activation function
# It converts the output into a value between 0 and 1.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# -----------------------------
# Input data
# -----------------------------

# We have 2 input features and 3 training examples.
X = np.array([
    [0.5, 0.8],
    [0.2, 0.4],
    [0.9, 0.7]
])


# -----------------------------
# Hidden layer parameters
# -----------------------------

# There are 3 neurons in the hidden layer.
#
# X has 2 features, so each hidden neuron needs
# 2 weights.
#
# Shape of W1 = (2, 3)
W1 = np.array([
    [0.2, 0.4, 0.6],
    [0.3, 0.5, 0.7]
])


# One bias for each hidden neuron
b1 = np.array([0.1, 0.1, 0.1])


# -----------------------------
# Output layer parameters
# -----------------------------

# The hidden layer has 3 neurons and the output layer
# has 1 neuron, so we need 3 weights.
#
# Shape of W2 = (3, 1)
W2 = np.array([
    [0.5],
    [0.6],
    [0.7]
])


# Bias for the output neuron
b2 = np.array([0.1])


# -----------------------------
# Forward Propagation
# -----------------------------

# First calculate the weighted sum for the hidden layer.
#
# Formula:
# Z1 = XW1 + b1
Z1 = np.dot(X, W1) + b1

print("Hidden Layer Weighted Sum (Z1):")
print(Z1)
print()


# Apply ReLU to the hidden layer.
#
# Formula:
# A1 = ReLU(Z1)
A1 = relu(Z1)

print("Hidden Layer Output (A1):")
print(A1)
print()


# Now send the hidden layer output to the output layer.
#
# Formula:
# Z2 = A1W2 + b2
Z2 = np.dot(A1, W2) + b2

print("Output Layer Weighted Sum (Z2):")
print(Z2)
print()


# Apply sigmoid to get the final output.
#
# Formula:
# A2 = Sigmoid(Z2)
A2 = sigmoid(Z2)

print("Final Output:")
print(A2)