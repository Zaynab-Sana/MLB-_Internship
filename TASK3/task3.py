# Module 14 - Introduction to Deep Learning
# Task 3 - Simple Neuron

import numpy as np


# Inputs

inputs = np.array([2, 3, 4])


# Weights

weights = np.array([0.5, 0.2, 0.1])


# Bias

bias = 1


# Calculate weighted values

weighted_values = inputs * weights


# Calculate weighted sum

weighted_sum = np.sum(weighted_values) + bias


# Apply ReLU activation function

output = max(0, weighted_sum)


# Print results

print("Inputs:", inputs)
print("Weights:", weights)
print("Bias:", bias)

print("\nWeighted values:", weighted_values)
print("Weighted sum:", weighted_sum)
print("Final output after ReLU:", output)