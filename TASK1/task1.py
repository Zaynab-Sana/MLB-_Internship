import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read Images
gray = cv2.imread("greyscale_image.jpg", cv2.IMREAD_GRAYSCALE)
rgb = cv2.imread("rgb_image.jpg")

if gray is None:
    print("Grayscale image not found!")
    exit()

if rgb is None:
    print("RGB image not found!")
    exit()

# Convert BGR to RGB as Open cv reads bgr image
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

noise_percentage = 5
noise_level = (noise_percentage / 100) * 255


# Add Gaussian Noise to Grayscale
height, width = gray.shape

noisy_gray = np.zeros((height, width), dtype=np.uint8)

for i in range(height):
    for j in range(width):

        noise = np.random.normal(0, noise_level)

        pixel = gray[i, j] + noise

        #Clipping
        if pixel < 0:
            pixel = 0
        elif pixel > 255:
            pixel = 255

        noisy_gray[i, j] = int(pixel)

# Add Gaussian Noise to RGB
height, width, channels = rgb.shape

noisy_rgb = np.zeros((height, width, channels), dtype=np.uint8)

for i in range(height):
    for j in range(width):
        for c in range(channels):

            # Generate Gaussian noise
            noise = np.random.normal(0, noise_level)

            # Add noise to pixel
            pixel = rgb[i, j, c] + noise

            # Clip pixel values manually
            if pixel < 0:
                pixel = 0
            elif pixel > 255:
                pixel = 255

            noisy_rgb[i, j, c] = int(pixel)


# Display Results
plt.figure(figsize=(12, 8))

# Original Grayscale
plt.subplot(2, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original Grayscale")
plt.axis("off")

# Noisy Grayscale
plt.subplot(2, 2, 2)
plt.imshow(noisy_gray, cmap="gray")
plt.title(f"Grayscale with {noise_percentage}% Gaussian Noise")
plt.axis("off")

# Original RGB
plt.subplot(2, 2, 3)
plt.imshow(rgb)
plt.title("Original RGB")
plt.axis("off")

# Noisy RGB
plt.subplot(2, 2, 4)
plt.imshow(noisy_rgb)
plt.title(f"RGB with {noise_percentage}% Gaussian Noise")
plt.axis("off")

plt.tight_layout()
plt.show()