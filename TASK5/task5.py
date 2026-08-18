# Module 14 - Introduction to Deep Learning
# Task 5 - Basic Neural Network Experiment

import numpy as np
import matplotlib.pyplot as plt


# Create a simple dataset

X = np.array([
    [0],
    [1],
    [2],
    [3],
    [4],
    [5]
], dtype=float)

y = np.array([
    [0],
    [2],
    [4],
    [6],
    [8],
    [10]
], dtype=float)


# Initialize weights and bias

np.random.seed(42)

weight = np.random.randn(1, 1)
bias = np.zeros((1, 1))


# Set learning rate and number of epochs

learning_rate = 0.01
epochs = 100


# Store loss values

losses = []


# Training

for epoch in range(epochs):

    # Forward pass

    prediction = np.dot(X, weight) + bias


    # Calculate Mean Squared Error

    error = prediction - y

    loss = np.mean(error ** 2)

    losses.append(loss)


    # Calculate gradients

    weight_gradient = (2 / len(X)) * np.dot(X.T, error)

    bias_gradient = (2 / len(X)) * np.sum(error)


    # Update weight and bias

    weight = weight - learning_rate * weight_gradient

    bias = bias - learning_rate * bias_gradient


    # Print progress

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}: "
            f"Loss = {loss:.4f}"
        )


# Final predictions

final_prediction = np.dot(X, weight) + bias


# Display results

print("\n" + "=" * 50)
print("TASK 5 — BASIC NEURAL NETWORK EXPERIMENT")
print("=" * 50)

print("\nInput:")
print(X.flatten())

print("\nActual Output:")
print(y.flatten())

print("\nFinal Prediction:")
print(final_prediction.flatten())

print("\nFinal Weight:")
print(weight)

print("\nFinal Bias:")
print(bias)

print("\nInitial Loss:")
print(losses[0])

print("Final Loss:")
print(losses[-1])


# Plot Loss vs Epochs

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    losses,
    linewidth=2
)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")

plt.grid(True)

plt.show()