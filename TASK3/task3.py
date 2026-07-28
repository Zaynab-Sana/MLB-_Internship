import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
image = cv2.imread("../TASK1/image.jpg")

if image is None:
    print("Image not found!")
    exit()

# Get image dimensions
height, width, channels = image.shape

# Sharpening Kernel
kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

# Output image
sharpened_image = np.zeros((height, width, channels), dtype=np.uint8)

# Manual Convolution
for i in range(1, height - 1):
    for j in range(1, width - 1):
        for c in range(channels):

            pixel_sum = 0

            # Apply 3×3 kernel manually
            for ki in range(-1, 2):
                for kj in range(-1, 2):
                    pixel_sum += (
                        int(image[i + ki, j + kj, c]) *
                        kernel[ki + 1][kj + 1]
                    )

            # Clipping
            if pixel_sum > 255:
                pixel_sum = 255
            elif pixel_sum < 0:
                pixel_sum = 0

            sharpened_image[i, j, c] = pixel_sum

# Convert BGR to RGB for Matplotlib
original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
sharpened_rgb = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)

# Display Images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(sharpened_rgb)
plt.title("Sharpened Image")
plt.axis("off")

plt.tight_layout()
plt.show()