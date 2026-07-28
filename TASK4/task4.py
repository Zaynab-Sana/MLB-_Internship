import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read Image
image = cv2.imread("../TASK1/image.jpg")

if image is None:
    print("Image not found!")
    exit()

# Convert BGR to RGB for display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

height, width, channels = image.shape

# Number of noisy pixels (5% of total pixels)
noise_percent = 0.05
num_pixels = int(height * width * noise_percent)

# Create copies
salt_image = image.copy()
pepper_image = image.copy()
salt_pepper_image = image.copy()
gaussian_image = image.copy()


# Salt Noise
for _ in range(num_pixels):
    x = np.random.randint(0, height)
    y = np.random.randint(0, width)

    for c in range(channels):
        salt_image[x, y, c] = 255


# Pepper Noise
for _ in range(num_pixels):
    x = np.random.randint(0, height)
    y = np.random.randint(0, width)

    for c in range(channels):
        pepper_image[x, y, c] = 0

# Salt & Pepper Noise
for _ in range(num_pixels):
    x = np.random.randint(0, height)
    y = np.random.randint(0, width)

    if np.random.rand() < 0.5:
        value = 255
    else:
        value = 0

    for c in range(channels):
        salt_pepper_image[x, y, c] = value

# Gaussian Noise
mean = 0
std = 25

for i in range(height):
    for j in range(width):
        for c in range(channels):

            noise = np.random.normal(mean, std)

            pixel = int(image[i, j, c]) + noise

            if pixel > 255:
                pixel = 255
            elif pixel < 0:
                pixel = 0

            gaussian_image[i, j, c] = int(pixel)

# Convert to RGB for display
salt_rgb = cv2.cvtColor(salt_image, cv2.COLOR_BGR2RGB)
pepper_rgb = cv2.cvtColor(pepper_image, cv2.COLOR_BGR2RGB)
salt_pepper_rgb = cv2.cvtColor(salt_pepper_image, cv2.COLOR_BGR2RGB)
gaussian_rgb = cv2.cvtColor(gaussian_image, cv2.COLOR_BGR2RGB)


# Display Results
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(salt_rgb)
plt.title("Salt Noise")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(pepper_rgb)
plt.title("Pepper Noise")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(salt_pepper_rgb)
plt.title("Salt & Pepper Noise")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(gaussian_rgb)
plt.title("Gaussian Noise")
plt.axis("off")

plt.tight_layout()
plt.show()