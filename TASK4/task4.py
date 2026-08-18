# Module 14 - Introduction to Deep Learning
# Task 4 - Dataset Split

import numpy as np


# Create a small dataset

data = np.array([
    [1, 10],
    [2, 20],
    [3, 30],
    [4, 40],
    [5, 50],
    [6, 60],
    [7, 70],
    [8, 80],
    [9, 90],
    [10, 100]
])


# Shuffle the dataset

np.random.seed(42)
np.random.shuffle(data)


# Calculate split sizes

total_samples = len(data)

train_size = int(0.7 * total_samples)
validation_size = int(0.15 * total_samples)


# Split the dataset

train_data = data[:train_size]

validation_data = data[
    train_size:train_size + validation_size
]

test_data = data[
    train_size + validation_size:
]


# Display the results

print("=" * 50)
print("TASK 4 — DATASET SPLIT")
print("=" * 50)

print("\nOriginal Dataset:")
print(data)

print("\nTraining Set:")
print(train_data)

print("\nValidation Set:")
print(validation_data)

print("\nTest Set:")
print(test_data)


# Display number of samples

print("\nNumber of samples:")
print("Total samples:", len(data))
print("Training samples:", len(train_data))
print("Validation samples:", len(validation_data))
print("Test samples:", len(test_data))


# Explain the purpose

print("\nPurpose of each dataset:")

print("""
Training Set:
Used to train the neural network and learn its weights
and biases.

Validation Set:
Used during development to check the model's performance
and help improve the model.

Test Set:
Used after training to evaluate the final performance
of the model on unseen data.
""")