import numpy as np


# --------------------------------------------------
# 1. Mean Squared Error
# --------------------------------------------------

def mse(y_true, y_pred):
    # Find the difference between actual and predicted values
    error = y_true - y_pred

    # Square the errors so positive and negative errors
    # don't cancel each other out
    squared_error = error ** 2

    # Take the average of all squared errors
    return np.mean(squared_error)


# --------------------------------------------------
# 2. Binary Cross-Entropy
# --------------------------------------------------

def binary_cross_entropy(y_true, y_pred):

    # Predictions of exactly 0 or 1 can cause problems
    # when we calculate log(0).
    #
    # So we keep predictions inside a very small safe range.
    epsilon = 1e-15

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Apply the Binary Cross-Entropy formula
    loss = -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )

    return loss


# --------------------------------------------------
# 3. Categorical Cross-Entropy
# --------------------------------------------------

def categorical_cross_entropy(y_true, y_pred):

    # Again, we avoid log(0) by keeping predictions
    # slightly above 0.
    epsilon = 1e-15

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # y_true contains one-hot encoded values.
    #
    # Example:
    # [0, 1, 0] means the second class is correct.
    #
    # Multiplying y_true with log(y_pred) means
    # we mainly take the probability of the correct class.
    loss = -np.mean(
        np.sum(y_true * np.log(y_pred), axis=1)
    )

    return loss


# --------------------------------------------------
# MSE Example
# --------------------------------------------------

y_true_mse = np.array([1, 0, 1])

y_pred_mse = np.array([0.8, 0.2, 0.6])

mse_loss = mse(y_true_mse, y_pred_mse)

print("MSE Loss:", mse_loss)


# --------------------------------------------------
# Binary Cross-Entropy Example
# --------------------------------------------------

y_true_binary = np.array([1, 0, 1])

y_pred_binary = np.array([0.9, 0.2, 0.8])

bce_loss = binary_cross_entropy(
    y_true_binary,
    y_pred_binary
)

print("Binary Cross-Entropy Loss:", bce_loss)


# --------------------------------------------------
# Categorical Cross-Entropy Example
# --------------------------------------------------

# We have three classes:
# Cat, Dog, Horse
#
# Actual classes:
# Sample 1 → Cat
# Sample 2 → Dog
# Sample 3 → Horse
#
# One-hot encoding:
# Cat   = [1, 0, 0]
# Dog   = [0, 1, 0]
# Horse = [0, 0, 1]

y_true_categorical = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])


# These are the model's predicted probabilities.
y_pred_categorical = np.array([
    [0.8, 0.1, 0.1],
    [0.2, 0.7, 0.1],
    [0.1, 0.2, 0.7]
])


cce_loss = categorical_cross_entropy(
    y_true_categorical,
    y_pred_categorical
)

print("Categorical Cross-Entropy Loss:", cce_loss)