import numpy as np


# -----------------------------
# Activation Function
# -----------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of sigmoid
# We need this during backpropagation.
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# -----------------------------
# Training Data
# -----------------------------

# We have two input features.
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)


# Target values
# This is the AND problem.
y = np.array([
    [0],
    [0],
    [0],
    [1]
], dtype=float)


# -----------------------------
# Initialize Parameters
# -----------------------------

np.random.seed(42)

# 2 input features -> 2 hidden neurons
W1 = np.random.randn(2, 2) * 0.5

# One bias for each hidden neuron
b1 = np.zeros((1, 2))


# 2 hidden neurons -> 1 output neuron
W2 = np.random.randn(2, 1) * 0.5

# One bias for the output neuron
b2 = np.zeros((1, 1))


# Learning rate controls how big
# each parameter update will be.
learning_rate = 0.5

epochs = 10000


# -----------------------------
# Training
# -----------------------------

for epoch in range(epochs):

    # ==================================
    # FORWARD PROPAGATION
    # ==================================

    # Input -> Hidden Layer
    Z1 = np.dot(X, W1) + b1

    # Apply sigmoid to hidden layer
    A1 = sigmoid(Z1)


    # Hidden Layer -> Output Layer
    Z2 = np.dot(A1, W2) + b2

    # Final prediction
    A2 = sigmoid(Z2)


    # ==================================
    # LOSS
    # ==================================

    # Small value used to avoid log(0)
    epsilon = 1e-15

    A2_safe = np.clip(A2, epsilon, 1 - epsilon)

    # Binary Cross-Entropy
    loss = -np.mean(
        y * np.log(A2_safe) +
        (1 - y) * np.log(1 - A2_safe)
    )


    # ==================================
    # BACKPROPAGATION
    # ==================================

    # Number of training examples
    m = len(X)


    # For sigmoid output + binary cross-entropy,
    # the derivative becomes simply:
    #
    # dZ2 = prediction - actual
    #
    # This is a very useful simplification.
    dZ2 = A2 - y


    # Gradient of output weights
    #
    # A1 is the input to the output layer.
    dW2 = np.dot(A1.T, dZ2) / m


    # Gradient of output bias
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m


    # Move the error back to the hidden layer.
    #
    # W2.T tells us how the output error
    # is distributed back to the hidden neurons.
    dA1 = np.dot(dZ2, W2.T)


    # We now pass this gradient through
    # the sigmoid activation used in the hidden layer.
    dZ1 = dA1 * sigmoid_derivative(Z1)


    # Gradient of hidden layer weights
    dW1 = np.dot(X.T, dZ1) / m


    # Gradient of hidden layer biases
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m


    # ==================================
    # UPDATE PARAMETERS
    # ==================================

    # Gradient Descent
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1


    # Print loss every 1000 epochs
    if (epoch + 1) % 1000 == 0:
        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {loss:.6f}"
        )


# -----------------------------
# Final Predictions
# -----------------------------

print("\nFinal Predictions:")

predictions = sigmoid(
    np.dot(
        sigmoid(np.dot(X, W1) + b1),
        W2
    ) + b2
)

print(predictions)


# Convert probabilities into 0 or 1
binary_predictions = (predictions >= 0.5).astype(int)

print("\nBinary Predictions:")
print(binary_predictions)

print("\nActual Values:")
print(y.astype(int))