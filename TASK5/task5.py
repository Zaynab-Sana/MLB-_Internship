import numpy as np


# Training data
# We want the model to learn the relationship:
# y = 2x + 1
X = np.array([1, 2, 3, 4, 5], dtype=float)

y = np.array([3, 5, 7, 9, 11], dtype=float)


# Start with random/incorrect parameters.
# We will let Gradient Descent improve them.
w = 0.0
b = 0.0


# Controls how big each update will be.
learning_rate = 0.01

# Number of times we train on the complete dataset.
epochs = 1000


for epoch in range(epochs):

    # -----------------------------
    # 1. Forward Pass
    # -----------------------------
    # Our model is:
    # prediction = w*x + b
    y_pred = w * X + b


    # -----------------------------
    # 2. Calculate Loss
    # -----------------------------
    # MSE tells us how far our predictions
    # are from the actual values.
    loss = np.mean((y - y_pred) ** 2)


    # -----------------------------
    # 3. Calculate Gradients
    # -----------------------------
    #
    # These formulas tell us how much
    # the loss changes when w and b change.
    #
    # We use the negative sign because our
    # error is written as (y - prediction).
    dw = (-2 / len(X)) * np.sum(X * (y - y_pred))

    db = (-2 / len(X)) * np.sum(y - y_pred)


    # -----------------------------
    # 4. Update Weight and Bias
    # -----------------------------
    #
    # We move in the opposite direction
    # of the gradient because we want to
    # decrease the loss.
    w = w - learning_rate * dw

    b = b - learning_rate * db


    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1}: "
            f"Loss = {loss:.4f}, "
            f"Weight = {w:.4f}, "
            f"Bias = {b:.4f}"
        )


# -----------------------------
# Final Results
# -----------------------------

print("\nFinal Weight:", w)
print("Final Bias:", b)


# Make predictions using the learned parameters
predictions = w * X + b

print("\nPredictions:", predictions)
print("Actual:", y)