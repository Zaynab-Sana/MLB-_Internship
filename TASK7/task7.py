import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read Image
image = cv2.imread("../TASK1/image.jpg")

if image is None:
    print("Image not found!")
    exit()

height, width, channels = image.shape

# Brightness Adjustment
brightness = 50
bright_image = np.zeros_like(image)

for i in range(height):
    for j in range(width):
        for c in range(channels):

            pixel = int(image[i, j, c]) + brightness

            if pixel > 255:
                pixel = 255
            elif pixel < 0:
                pixel = 0

            bright_image[i, j, c] = pixel


# Contrast Adjustment
alpha = 1.5
beta = 20

contrast_image = np.zeros_like(image)

for i in range(height):
    for j in range(width):
        for c in range(channels):

            pixel = alpha * int(image[i, j, c]) + beta

            if pixel > 255:
                pixel = 255
            elif pixel < 0:
                pixel = 0

            contrast_image[i, j, c] = int(pixel)


# Sharpening
kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

sharpen_image = np.zeros_like(image)

for i in range(1, height - 1):
    for j in range(1, width - 1):
        for c in range(channels):

            total = 0

            for ki in range(-1, 2):
                for kj in range(-1, 2):

                    total += (
                        int(image[i + ki, j + kj, c])
                        * kernel[ki + 1][kj + 1]
                    )

            if total > 255:
                total = 255
            elif total < 0:
                total = 0

            sharpen_image[i, j, c] = total


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


# Median Filter
denoise_image = np.zeros_like(image)

for i in range(1, height - 1):
    for j in range(1, width - 1):
        for c in range(channels):

            values = []

            for ki in range(-1, 2):
                for kj in range(-1, 2):

                    values.append(
                        int(noisy_image[i + ki, j + kj, c])
                    )

            values.sort()

            denoise_image[i, j, c] = values[4]


# Convert Images to RGB
original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
bright_rgb = cv2.cvtColor(bright_image, cv2.COLOR_BGR2RGB)
contrast_rgb = cv2.cvtColor(contrast_image, cv2.COLOR_BGR2RGB)
sharpen_rgb = cv2.cvtColor(sharpen_image, cv2.COLOR_BGR2RGB)
noisy_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
denoise_rgb = cv2.cvtColor(denoise_image, cv2.COLOR_BGR2RGB)


# Display Images
plt.figure(figsize=(18, 10))

plt.subplot(2, 3, 1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(bright_rgb)
plt.title("Brightness Adjustment")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(contrast_rgb)
plt.title("Contrast Enhancement")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(sharpen_rgb)
plt.title("Sharpening")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(noisy_rgb)
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(denoise_rgb)
plt.title("Denoised (Median Filter)")
plt.axis("off")

plt.tight_layout()
plt.show()

# ==================================
# Observations
# ==================================
print("\nImage Enhancement Comparison")
print("-------------------------------------")
print("1. Brightness Adjustment:")
print("   Increased overall light intensity of the image.")

print("\n2. Contrast Enhancement:")
print("   Increased the difference between dark and bright regions.")

print("\n3. Sharpening:")
print("   Enhanced edges and fine details.")

print("\n4. Denoising:")
print("   Median filter removed Salt & Pepper noise while preserving edges.")