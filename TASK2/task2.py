import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read Grayscale Image
image = cv2.imread("../TASK1/greyscale_image.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found!")
    exit()


# Add 5% Gaussian Noise
noise_percentage = 2
noise_level = (noise_percentage / 100) * 255

height, width = image.shape

noisy = np.zeros((height, width), dtype=np.uint8)

for i in range(height):
    for j in range(width):

        noise = np.random.normal(0, noise_level)

        pixel = image[i, j] + noise

        if pixel < 0:
            pixel = 0
        elif pixel > 255:
            pixel = 255

        noisy[i, j] = int(pixel)


# Mean Filter
kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])

mean_filtered = np.zeros((height, width), dtype=np.uint8)

for i in range(1, height - 1):
    for j in range(1, width - 1):

        region = noisy[i-1:i+2, j-1:j+2]

        value = np.sum(region * kernel) / 9

        if value < 0:
            value = 0
        elif value > 255:
            value = 255

        mean_filtered[i, j] = int(value)


# Gaussian Filter (Manual)
gaussian_kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
])

gaussian_filtered = np.zeros((height, width), dtype=np.uint8)

for i in range(1, height - 1):
    for j in range(1, width - 1):

        region = noisy[i-1:i+2, j-1:j+2]

        value = np.sum(region * gaussian_kernel) / 16

        if value < 0:
            value = 0
        elif value > 255:
            value = 255

        gaussian_filtered[i, j] = int(value)


# Display Results
plt.figure(figsize=(14,5))

plt.subplot(1,4,1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(noisy, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(mean_filtered, cmap="gray")
plt.title("Mean Filter")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(gaussian_filtered, cmap="gray")
plt.title("Gaussian Filter")
plt.axis("off")

plt.tight_layout()

# -----------------------------------
# Task 2 Explanation
# -----------------------------------

print("\n========== Task 2 Analysis ==========\n")

print("1. Mean Filter")
print("- The Mean Filter replaces each pixel with the average of its 3×3 neighborhood.")
print("- All neighboring pixels have equal weight.")
print("- It removes Gaussian noise but also blurs edges and fine details.\n")

print("2. Gaussian Filter")
print("- The Gaussian Filter uses a weighted 3×3 kernel.")
print("- The center pixel has the highest weight, while surrounding pixels have smaller weights.")
print("- It removes Gaussian noise while preserving edges better than the Mean Filter.\n")

print("Comparison:")
print("- Mean Filter: More smoothing but greater loss of image details.")
print("- Gaussian Filter: Better balance between noise removal and edge preservation.")
print("- For Gaussian noise, the Gaussian Filter generally performs better than the Mean Filter.\n")

print("Conclusion:")
print("The Gaussian Filter is the preferred choice for removing Gaussian noise because")
print("it reduces noise effectively while maintaining important image features and edges.")
plt.show()