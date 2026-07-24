import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

# ========================== MEDIAN FILTER =========================
# Read grayscale image
image = cv2.imread("../TASK1/image.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found!")
    exit()

# Image dimensions
height, width = image.shape

# Create noisy image
noisy_image = image.copy()

# Noise percentage
noise_percentage = 0.05

# Number of noisy pixels
total_pixels = height * width
num_noise_pixels = int(total_pixels * noise_percentage)

# Add Salt and Pepper Noise manually
for _ in range(num_noise_pixels):

    row = random.randint(0, height - 1)
    col = random.randint(0, width - 1)

    if random.random() < 0.5:
        noisy_image[row, col] = 0      # Pepper
    else:
        noisy_image[row, col] = 255    # Salt

# ================= Manual Median Filter =================
#create output image
output = noisy_image.copy()

#loop for applying median filter
for row in range(1, height - 1):
    for col in range(1, width - 1):

        #store neighbour elements in array
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                neighbors.append(noisy_image[row + i, col + j])

        # Sort values
        neighbors.sort()
        # Calculate the median
        median_value = np.median(neighbors)
        # Replace center pixel
        output[row, col] = median_value

# ================= Display Images =================

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(noisy_image, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(output, cmap="gray")
plt.title("Median Filter Output")
plt.axis("off")

plt.tight_layout()

# Save image
success = cv2.imwrite("median_filter_image.jpg", output)
cv2.imwrite("noisy_image.jpg", noisy_image)

plt.show()

if success:
    print("Image saved successfully!")
else:
    print("Failed to save image.")