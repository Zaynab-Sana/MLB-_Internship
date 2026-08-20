import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms


# --------------------------------------------------
# 1. LOAD IMAGE
# --------------------------------------------------

# Change this to your image path
image = Image.open("image.jpg").convert("RGB")


# --------------------------------------------------
# 2. DEFINE AUGMENTATIONS
# --------------------------------------------------

# Horizontal flip
horizontal_flip = transforms.RandomHorizontalFlip(
    p=1.0
)

# Rotate image by 30 degrees
rotation = transforms.RandomRotation(
    degrees=(30, 30)
)

# Random crop
crop = transforms.RandomCrop(
    size=(150, 150)
)

# Resize image
scaling = transforms.Resize(
    size=(300, 300)
)

# Change brightness
brightness = transforms.ColorJitter(
    brightness=0.5
)

# Change contrast
contrast = transforms.ColorJitter(
    contrast=0.5
)


# --------------------------------------------------
# 3. APPLY AUGMENTATIONS
# --------------------------------------------------

flipped_image = horizontal_flip(image)

rotated_image = rotation(image)

cropped_image = crop(image)

scaled_image = scaling(image)

bright_image = brightness(image)

contrast_image = contrast(image)


# --------------------------------------------------
# 4. DISPLAY RESULTS
# --------------------------------------------------

plt.figure(figsize=(12, 8))


# Original
plt.subplot(2, 4, 1)
plt.imshow(image)
plt.title("Original")
plt.axis("off")


# Flip
plt.subplot(2, 4, 2)
plt.imshow(flipped_image)
plt.title("Flipped")
plt.axis("off")


# Rotation
plt.subplot(2, 4, 3)
plt.imshow(rotated_image)
plt.title("Rotated")
plt.axis("off")


# Crop
plt.subplot(2, 4, 4)
plt.imshow(cropped_image)
plt.title("Cropped")
plt.axis("off")


# Scaling
plt.subplot(2, 4, 5)
plt.imshow(scaled_image)
plt.title("Scaled")
plt.axis("off")


# Brightness
plt.subplot(2, 4, 6)
plt.imshow(bright_image)
plt.title("Brightness")
plt.axis("off")


# Contrast
plt.subplot(2, 4, 7)
plt.imshow(contrast_image)
plt.title("Contrast")
plt.axis("off")


plt.tight_layout()
plt.show()