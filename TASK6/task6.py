import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read Image
image = cv2.imread("../TASK1/image.jpg")

if image is None:
    print("Image not found!")
    exit()

height, width, channels = image.shape


# Blur Kernel
blur_kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])


# Sharpening Kernel
sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])


# Output Images
blurred_image = np.zeros_like(image)
restored_image = np.zeros_like(image)

#loop for blurring image
for i in range(1, height - 1):
    for j in range(1, width - 1):
        for c in range(channels):

            total = 0

            for ki in range(-1, 2):
                for kj in range(-1, 2):

                    pixel = int(image[i + ki, j + kj, c])
                    total += pixel * blur_kernel[ki + 1][kj + 1]

            blurred_image[i, j, c] = total // 9

#loop for sharpening image
for i in range(1, height - 1):
    for j in range(1, width - 1):
        for c in range(channels):

            total = 0

            for ki in range(-1, 2):
                for kj in range(-1, 2):

                    pixel = int(blurred_image[i + ki, j + kj, c])
                    weight = sharpen_kernel[ki + 1][kj + 1]

                    total += pixel * weight

            if total > 255:
                total = 255
            elif total < 0:
                total = 0

            restored_image[i, j, c] = total


original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
blurred_rgb = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
restored_rgb = cv2.cvtColor(restored_image, cv2.COLOR_BGR2RGB)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(blurred_rgb)
plt.title("Blurred Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(restored_rgb)
plt.title("Restored (Sharpened)")
plt.axis("off")

plt.tight_layout()
plt.show()