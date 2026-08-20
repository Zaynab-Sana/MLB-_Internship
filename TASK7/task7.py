import torch
import torch.nn as nn
import torch.optim as optim


# --------------------------------------------------
# Training data
# --------------------------------------------------

# Each row contains two input features: x1 and x2.
#
# We are teaching the network the AND operation:
#
# 0 AND 0 = 0
# 0 AND 1 = 0
# 1 AND 0 = 0
# 1 AND 1 = 1

X = torch.tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=torch.float32)


# Actual answers for each input
y = torch.tensor([
    [0],
    [0],
    [0],
    [1]
], dtype=torch.float32)


# --------------------------------------------------
# Neural Network
# --------------------------------------------------

class SimpleNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        # First layer:
        # 2 input features -> 4 hidden neurons
        self.hidden = nn.Linear(2, 4)

        # Output layer:
        # 4 hidden neurons -> 1 output
        self.output = nn.Linear(4, 1)

        # ReLU is commonly used in hidden layers.
        self.relu = nn.ReLU()

        # Sigmoid converts the final output into
        # a probability between 0 and 1.
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        # Input passes through the hidden layer
        x = self.hidden(x)

        # Apply ReLU activation
        x = self.relu(x)

        # Send hidden-layer output to output layer
        x = self.output(x)

        # Convert final value into a probability
        x = self.sigmoid(x)

        return x


# Create the neural network
model = SimpleNeuralNetwork()


# --------------------------------------------------
# Loss Function
# --------------------------------------------------

# Binary Cross-Entropy is suitable for
# binary classification.
criterion = nn.BCELoss()


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

# Adam automatically updates the model's
# weights and biases using the calculated gradients.
optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)


# --------------------------------------------------
# Training
# --------------------------------------------------

epochs = 5000

for epoch in range(epochs):

    # -----------------------------
    # 1. Forward Pass
    # -----------------------------

    # Get predictions from the network
    predictions = model(X)


    # -----------------------------
    # 2. Calculate Loss
    # -----------------------------

    loss = criterion(predictions, y)


    # -----------------------------
    # 3. Clear old gradients
    # -----------------------------

    # Gradients are accumulated by PyTorch,
    # so we clear them before calculating new ones.
    optimizer.zero_grad()


    # -----------------------------
    # 4. Backpropagation
    # -----------------------------

    # PyTorch automatically calculates
    # all required gradients.
    loss.backward()


    # -----------------------------
    # 5. Update Parameters
    # -----------------------------

    # The optimizer uses the gradients
    # to update the weights and biases.
    optimizer.step()


    # Print progress
    if (epoch + 1) % 500 == 0:
        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {loss.item():.6f}"
        )


# --------------------------------------------------
# Testing the trained model
# --------------------------------------------------

print("\nFinal Predictions:")

with torch.no_grad():

    predictions = model(X)

    print(predictions)


# Convert probabilities into 0 or 1
binary_predictions = (predictions >= 0.5).float()

print("\nBinary Predictions:")
print(binary_predictions)


print("\nActual Values:")
print(y)