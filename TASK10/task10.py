import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# --------------------------------------------------
# 1. DEVICE
# --------------------------------------------------

# Use GPU if available, otherwise use CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)


# --------------------------------------------------
# 2. IMAGE TRANSFORMS
# --------------------------------------------------

# Convert images into the format expected by ResNet
transform = transforms.Compose([

    # Resize image
    transforms.Resize((224, 224)),

    # Convert image to PyTorch tensor
    transforms.ToTensor(),

    # Normalize using ImageNet values
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# 3. LOAD DATASET
# --------------------------------------------------

train_dataset = datasets.ImageFolder(
    "dataset/train",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    "dataset/test",
    transform=transform
)


# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


# Number of classes
num_classes = len(train_dataset.classes)

print("Classes:", train_dataset.classes)
print("Number of classes:", num_classes)


# --------------------------------------------------
# 4. LOAD PRETRAINED RESNET18
# --------------------------------------------------

# Load ResNet18 with pretrained ImageNet weights
model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)


# --------------------------------------------------
# 5. FREEZE PRETRAINED LAYERS
# --------------------------------------------------

# Do not update existing ResNet weights
for param in model.parameters():
    param.requires_grad = False


# --------------------------------------------------
# 6. REPLACE CLASSIFICATION HEAD
# --------------------------------------------------

# Get number of inputs to the original final layer
num_features = model.fc.in_features


# Replace original classifier
# with a classifier for our classes
model.fc = nn.Linear(
    num_features,
    num_classes
)


# Move model to CPU/GPU
model = model.to(device)


# --------------------------------------------------
# 7. LOSS FUNCTION
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# 8. OPTIMIZER
# --------------------------------------------------

# Only train the new classification layer
optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001
)


# --------------------------------------------------
# 9. TRAINING
# --------------------------------------------------

epochs = 5

for epoch in range(epochs):

    # Training mode
    model.train()

    total_loss = 0

    for images, labels in train_loader:

        # Move data to device
        images = images.to(device)
        labels = labels.to(device)

        # Remove old gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update classification layer
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {total_loss / len(train_loader):.4f}"
    )


# --------------------------------------------------
# 10. TEST MODEL
# --------------------------------------------------

model.eval()

correct = 0
total = 0


# No gradients are needed during testing
with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Get predictions
        outputs = model(images)

        # Select class with highest score
        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


# Calculate accuracy
accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")
