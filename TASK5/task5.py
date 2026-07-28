import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read Image
image = cv2.imread("../TASK1/image.jpg")

if image is None:
    print("Image not found!")
    exit()

height, width, channels = image.shape


# Add Salt & Pepper Noise
noisy_image = image.copy()

noise_percent = 0.05
num_pixels = int(height * width * noise_percent)

for _ in range(num_pixels):
    x = np.random.randint(0, height)
    y = np.random.randint(0, width)

    if np.random.rand() < 0.5:
        value = 255
    else:
        value = 0

    for c in range(channels):
        noisy_image[x, y, c] = value


# Output Images
mean_image = np.zeros_like(image)
median_image = np.zeros_like(image)
gaussian_image = np.zeros_like(image)

# Gaussian Kernel
gaussian_kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
])


# Manual Mean Filter
for i in range(1, height - 1):
    for j in range(1, width - 1):
        for c in range(channels):

            total = 0

            for ki in range(-1, 2):
                for kj in range(-1, 2):
                    total += int(noisy_image[i + ki, j + kj, c])

            mean_image[i, j, c] = total // 9


# Manual Median Filter
for i in range(1, height - 1):
    for j in range(1, width - 1):
        for c in range(channels):

            values = []

            for ki in range(-1, 2):
                for kj in range(-1, 2):
                    values.append(int(noisy_image[i + ki, j + kj, c]))

            values.sort()

            median_image[i, j, c] = values[4]


# Manual Gaussian Filter
for i in range(1, height - 1):
    for j in range(1, width - 1):
        for c in range(channels):

            total = 0

            for ki in range(-1, 2):
                for kj in range(-1, 2):
                    pixel = int(noisy_image[i + ki, j + kj, c])
                    weight = gaussian_kernel[ki + 1, kj + 1]
                    total += pixel * weight

            gaussian_image[i, j, c] = total // 16


# Convert to RGB
original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
noisy_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
mean_rgb = cv2.cvtColor(mean_image, cv2.COLOR_BGR2RGB)
median_rgb = cv2.cvtColor(median_image, cv2.COLOR_BGR2RGB)
gaussian_rgb = cv2.cvtColor(gaussian_image, cv2.COLOR_BGR2RGB)


# Display Images
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(noisy_rgb)
plt.title("Salt & Pepper Noise")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(mean_rgb)
plt.title("Mean Filter")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(median_rgb)
plt.title("Median Filter")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(gaussian_rgb)
plt.title("Gaussian Filter")
plt.axis("off")

plt.tight_layout()
plt.show()