"""
model.py
--------
Defines SimpleCNN: a Convolutional Neural Network built entirely from
scratch with plain PyTorch layers. No pretrained weights, no ResNet/VGG/
MobileNet/EfficientNet backbone -- every number in this network starts
random and is learned only from the TrashNet images.

--------------------------------------------------------------------
CNN CONCEPTS IN PLAIN LANGUAGE (read this before the architecture)
--------------------------------------------------------------------
- Convolution: sliding a small window (the "kernel") over the image and,
  at each position, multiplying the pixels under the window by learned
  weights and summing them up. Think of it as a small stamp that lights
  up when it sees a pattern it recognizes (an edge, a corner, a texture).

- Kernel / filter: the small grid of learnable numbers used in a
  convolution (e.g. 3x3). Each filter learns to detect one specific
  pattern. A conv layer with 32 filters produces 32 different "pattern
  detectors" applied to the same image.

- Feature map: the output of applying one filter across the whole image.
  It's a 2D grid showing WHERE that pattern was found and how strongly.

- ReLU (Rectified Linear Unit): replaces every negative number with 0
  and keeps positive numbers unchanged: f(x) = max(0, x). This adds
  non-linearity -- without it, stacking many conv layers would still
  only be able to represent simple straight-line relationships, no
  matter how many layers you add.

- Pooling (we use MaxPool): shrinks a feature map by keeping only the
  strongest activation in each small window (e.g. 2x2). This reduces
  the amount of data flowing through the network, makes the model care
  less about the EXACT pixel position of a pattern (translation
  tolerance), and reduces overfitting.

- Flattening: after the conv+pool blocks, we have a 3D tensor (channels
  x height x width). Flatten reshapes it into a single 1D vector so it
  can be fed into a normal fully-connected layer.

- Fully connected (Linear) layer: every input number is connected to
  every output number through a learned weight. This is where the
  network combines all the detected features together to make the
  final decision.

- Softmax: converts raw output numbers ("logits") into probabilities
  that sum to 1 across the 6 classes. We do NOT apply Softmax manually
  inside the model -- see the note in the loss function section of
  train.py for why.

- Why CNNs suit image classification: images have local structure
  (nearby pixels are related) and patterns that can appear anywhere in
  the frame (a plastic bottle could be top-left or bottom-right). A
  convolution's sliding-window design directly exploits both: it looks
  at local neighborhoods, and it reuses the SAME filter everywhere in
  the image, so it detects a pattern no matter where it appears.
--------------------------------------------------------------------

ARCHITECTURE

    Input (3 x 128 x 128)
      -> Conv2d(3, 32, k=3, pad=1)  -> ReLU -> MaxPool(2)   => 32 x 64 x 64
      -> Conv2d(32, 64, k=3, pad=1) -> ReLU -> MaxPool(2)   => 64 x 32 x 32
      -> Conv2d(64, 128, k=3, pad=1)-> ReLU -> MaxPool(2)   => 128 x 16 x 16
      -> Flatten                                            => 32768
      -> Linear(32768, 256) -> ReLU -> Dropout(p)
      -> Linear(256, 6)                                     => 6 class scores (logits)

Dimension math for each conv block (padding=1, stride=1, kernel=3
keeps height/width unchanged; MaxPool(2) halves height/width):
    128 -> conv (still 128) -> pool -> 64
    64  -> conv (still 64)  -> pool -> 32
    32  -> conv (still 32)  -> pool -> 16
Final feature map: 128 channels x 16 x 16 = 32768 values, which is
exactly the input size of the first Linear layer below.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 6, num_filters: int = 32, dropout: float = 0.5):
        """
        num_classes: number of output categories (6 for TrashNet).
        num_filters: number of filters in the FIRST conv layer. The
                     second and third conv layers use 2x and 4x this
                     value. Exposed as a parameter so it can be tuned
                     in the Day-3 hyperparameter experiments.
        dropout:     probability of zeroing a neuron in the fully
                     connected layer during training, used to reduce
                     overfitting. Also exposed for experimentation.
        """
        super().__init__()

        f1, f2, f3 = num_filters, num_filters * 2, num_filters * 4

        # --- Convolutional feature extractor ---
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=f1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 128 -> 64
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=f1, out_channels=f2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 64 -> 32
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=f2, out_channels=f3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 32 -> 16
        )

        self.flatten = nn.Flatten()

        # Feature map after conv_block3 is (f3 channels) x 16 x 16.
        flattened_size = f3 * 16 * 16

        # --- Fully connected classifier head ---
        self.fc1 = nn.Linear(flattened_size, 256)
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines exactly what happens to one batch of images as it
        passes through the network, step by step.

        x starts as shape: (batch_size, 3, 128, 128)
        """
        x = self.conv_block1(x)   # -> (batch, 32, 64, 64)
        x = self.conv_block2(x)   # -> (batch, 64, 32, 32)
        x = self.conv_block3(x)   # -> (batch, 128, 16, 16)

        x = self.flatten(x)       # -> (batch, 128*16*16) = (batch, 32768)

        x = self.fc1(x)           # -> (batch, 256)
        x = self.relu_fc(x)
        x = self.dropout(x)       # randomly zeroes some of the 256 values (training only)

        x = self.fc2(x)           # -> (batch, 6)  <-- raw logits, NOT probabilities
        return x


def build_model(num_classes: int = 6, num_filters: int = 32, dropout: float = 0.5,
                 device=None) -> SimpleCNN:
    """Convenience constructor used by train.py / evaluate.py / predict.py."""
    model = SimpleCNN(num_classes=num_classes, num_filters=num_filters, dropout=dropout)
    if device is not None:
        model = model.to(device)
    return model


if __name__ == "__main__":
    # Quick sanity check: run running a fake batch through the network
    # and print the shape after every stage, confirming the dimension
    # math in the docstring above is correct.
    model = SimpleCNN()
    dummy_input = torch.randn(4, 3, 128, 128)  # batch of 4 fake RGB images

    x = dummy_input
    print("Input:            ", x.shape)
    x = model.conv_block1(x)
    print("After conv_block1:", x.shape)
    x = model.conv_block2(x)
    print("After conv_block2:", x.shape)
    x = model.conv_block3(x)
    print("After conv_block3:", x.shape)
    x = model.flatten(x)
    print("After flatten:    ", x.shape)
    x = model.fc1(x)
    print("After fc1:        ", x.shape)
    x = model.fc2(x)
    print("After fc2 (logits):", x.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal trainable parameters: {total_params:,}")
