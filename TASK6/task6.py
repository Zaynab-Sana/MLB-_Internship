import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read grayscale image
image = cv2.imread("image3.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found!")
    exit()

# Get image dimensions
height, width = image.shape

# Create output image
output = np.zeros((height, width), dtype=np.uint8)

# 3x3 Blur Kernel
kernel = [
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
]

# OUTER LOOP: Move kernel over image
for row in range(1, height - 1):
    for col in range(1, width - 1):

        total = 0

        # Visit all 9 pixels inside the kernel
        for i in range(-1, 2):
            for j in range(-1, 2):

                pixel = image[row + i][col + j]
                kernel_value = kernel[i + 1][j + 1]

                total = total + (pixel * kernel_value)

        # Keep pixel value between 0 and 255
        if total < 0:
            total = 0

        if total > 255:
            total = 255

        output[row][col] = int(total)

# Display images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output, cmap="gray")
plt.title("After Manual Convolution")
plt.axis("off")

plt.show()

# Save output
cv2.imwrite("manual_convolution.jpg", output)