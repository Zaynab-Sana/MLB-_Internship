import numpy as np


class Perceptron:

    def __init__(self, learning_rate=0.1, epochs=10):
        # Learning rate decides how much we change the weights after a mistake
        self.learning_rate = learning_rate

        # Epoch means how many times we go through the complete training data
        self.epochs = epochs

        self.weights = None
        self.bias = 0

    def activation(self, z):
        # The perceptron uses a simple step function.
        # If the calculated value is 0 or greater, we predict 1.
        # Otherwise, we predict 0.
        return 1 if z >= 0 else 0

    def predict(self, X):
        # np.dot() calculates the weighted sum of the inputs.
        # Basically: x1*w1 + x2*w2 + ... + bias
        z = np.dot(X, self.weights) + self.bias

        # Apply the step function to every calculated value
        return np.array([self.activation(value) for value in z])

    def fit(self, X, y):

        # Start all weights from zero.
        # X.shape[1] tells us how many input features we have.
        self.weights = np.zeros(X.shape[1])

        # Start the bias from zero as well
        self.bias = 0

        # Repeat the training process for the given number of epochs
        for epoch in range(self.epochs):

            # Take one training example at a time
            for i in range(len(X)):

                # Calculate the weighted sum:
                # z = x1*w1 + x2*w2 + bias
                z = np.dot(X[i], self.weights) + self.bias

                # Convert the weighted sum into either 0 or 1
                prediction = self.activation(z)

                # Error tells us how far our prediction was from the actual answer
                # If prediction is correct, error will be 0
                error = y[i] - prediction

                # Update the weights using the perceptron learning rule:
                # new weight = old weight + learning_rate * error * input
                self.weights += self.learning_rate * error * X[i]

                # Update the bias in the same way, but without multiplying by input
                self.bias += self.learning_rate * error

            # Show how the weights and bias are changing after each epoch
            print(f"Epoch {epoch + 1}:")
            print("Weights:", self.weights)
            print("Bias:", self.bias)
            print()


# Training data
# We are teaching the perceptron to behave like an AND gate.
#
# 0 AND 0 = 0
# 0 AND 1 = 0
# 1 AND 0 = 0
# 1 AND 1 = 1

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Correct outputs for the above inputs
y = np.array([0, 0, 0, 1])


# Create our perceptron
model = Perceptron(learning_rate=0.1, epochs=10)


# Train the perceptron
# During training, it will make predictions and update
# the weights and bias whenever it makes a mistake.
model.fit(X, y)


# Test the trained perceptron
predictions = model.predict(X)


# Compare what the model predicted with the correct answers
print("Predictions:", predictions)
print("Actual:", y)