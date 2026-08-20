import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Same random weights every time
torch.manual_seed(42)


# ---------------- DATA ----------------

# 2 input features
X = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])

# Correct answers
y = torch.tensor([
    [0.],
    [1.],
    [1.],
    [0.]
])


# ---------------- NEURAL NETWORK ----------------

class SimpleNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            # 2 inputs -> 8 neurons
            nn.Linear(2, 8),

            # Adds non-linearity
            nn.ReLU(),

            # 8 neurons -> 1 output
            nn.Linear(8, 1),

            # Output between 0 and 1
            nn.Sigmoid()
        )

    def forward(self, x):

        # Send input through the network
        return self.network(x)


# ---------------- TRAINING FUNCTION ----------------

def train_model(optimizer_name):

    # Create a fresh model
    model = SimpleNN()

    # Measures prediction error
    criterion = nn.BCELoss()


    # ---------------- OPTIMIZER ----------------

    if optimizer_name == "SGD":

        # Basic gradient descent
        optimizer = optim.SGD(
            model.parameters(),
            lr=0.1
        )

    elif optimizer_name == "Momentum":

        # SGD + memory of previous updates
        optimizer = optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9
        )

    elif optimizer_name == "Adam":

        # Adaptive optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=0.01
        )


    losses = []


    # ---------------- TRAINING ----------------

    for epoch in range(1000):

        # 1. Forward propagation
        # Input -> Network -> Prediction
        predictions = model(X)


        # 2. Calculate loss
        # Compare prediction with actual answer
        loss = criterion(predictions, y)


        # 3. Remove previous gradients
        # PyTorch otherwise adds old + new gradients
        optimizer.zero_grad()


        # 4. Backpropagation
        # Calculate gradients using chain rule
        loss.backward()


        # 5. Update weights and biases
        # Optimizer uses gradients to reduce loss
        optimizer.step()


        # Save loss for graph
        losses.append(loss.item())


    return model, losses


# ---------------- TRAIN 3 MODELS ----------------

# Train with normal SGD
sgd_model, sgd_losses = train_model("SGD")

# Train with Momentum
momentum_model, momentum_losses = train_model("Momentum")

# Train with Adam
adam_model, adam_losses = train_model("Adam")


# ---------------- COMPARE ----------------

plt.plot(sgd_losses, label="SGD")
plt.plot(momentum_losses, label="Momentum")
plt.plot(adam_losses, label="Adam")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Optimizer Comparison")

plt.legend()
plt.grid()
plt.show()


# ---------------- PREDICTIONS ----------------

print("SGD:")
print(sgd_model(X).detach())

print("Momentum:")
print(momentum_model(X).detach())

print("Adam:")
print(adam_model(X).detach())