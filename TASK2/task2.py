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

# Contrast parameters
alpha = 1.5      # Contrast
beta = 60        # Brightness

# Create output image
output_image = np.zeros((height, width, channels), dtype=np.uint8)

# Contrast Adjustment
for i in range(height):
    for j in range(width):
        for k in range(channels):

            new_pixel = alpha * int(image[i, j, k]) + beta

            # Clipping
            if new_pixel > 255:
                new_pixel = 255
            elif new_pixel < 0:
                new_pixel = 0

            output_image[i, j, k] = int(new_pixel)

# Convert BGR to RGB for Matplotlib
original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

# Display images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output_rgb)
plt.title(f"Contrast Adjusted\nα={alpha}, β={beta}")
plt.axis("off")

plt.tight_layout()
plt.show()