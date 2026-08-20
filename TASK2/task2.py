import numpy as np
import matplotlib.pyplot as plt


# Sigmoid converts any value into a number between 0 and 1.
# It is commonly used for binary classification.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Tanh converts values into a range between -1 and 1.
# Unlike sigmoid, it is centered around zero.
def tanh(x):
    return np.tanh(x)


# ReLU keeps positive values as they are
# and changes negative values to zero.
def relu(x):
    return np.maximum(0, x)


# Leaky ReLU works like ReLU for positive values,
# but allows a small negative value for negative inputs.
# This helps reduce the "dying ReLU" problem.
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


# Softmax converts a group of raw scores into probabilities.
# All probabilities will be between 0 and 1
# and their total will be equal to 1.
def softmax(x):

    # Subtracting the maximum value makes the calculation
    # more numerically stable when the input values are large.
    exp_values = np.exp(x - np.max(x))

    return exp_values / np.sum(exp_values)


# Create input values from -10 to 10.
# These values will be passed through the activation functions.
x = np.linspace(-10, 10, 400)


# Calculate the output of each activation function.
sigmoid_values = sigmoid(x)
tanh_values = tanh(x)
relu_values = relu(x)
leaky_relu_values = leaky_relu(x)


# Plot Sigmoid
plt.figure(figsize=(8, 5))
plt.plot(x, sigmoid_values)
plt.title("Sigmoid Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()


# Plot Tanh
plt.figure(figsize=(8, 5))
plt.plot(x, tanh_values)
plt.title("Tanh Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()


# Plot ReLU
plt.figure(figsize=(8, 5))
plt.plot(x, relu_values)
plt.title("ReLU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()


# Plot Leaky ReLU
plt.figure(figsize=(8, 5))
plt.plot(x, leaky_relu_values)
plt.title("Leaky ReLU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()


# Example scores for Softmax
scores = np.array([2.0, 1.0, 0.1])

softmax_values = softmax(scores)

print("Softmax Input:", scores)
print("Softmax Output:", softmax_values)
print("Sum of probabilities:", np.sum(softmax_values))