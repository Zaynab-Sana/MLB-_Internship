import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Read Image
# ==========================================
image = cv2.imread("../TASK1/image.jpg")

if image is None:
    print("Image not found!")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

height, width, channels = image.shape

# ==========================================
# Manual RGB to Gray
# ==========================================
gray = np.zeros((height, width), dtype=np.float32)

for i in range(height):
    for j in range(width):

        r = image[i, j, 0]
        g = image[i, j, 1]
        b = image[i, j, 2]

        gray[i, j] = (
            0.299 * r +
            0.587 * g +
            0.114 * b
        )

# ==========================================
# Sobel Kernels
# ==========================================
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)

# ==========================================
# Laplacian Kernel
# 4-Neighbour Kernel
# ==========================================
laplacian_kernel = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
], dtype=np.float32)

# ==========================================
# Output Images
# ==========================================
gx = np.zeros((height, width), dtype=np.float32)
gy = np.zeros((height, width), dtype=np.float32)

sobel = np.zeros((height, width), dtype=np.float32)

laplacian = np.zeros((height, width), dtype=np.float32)

# ==========================================
# Manual Convolution
# ==========================================
for i in range(1, height - 1):
    for j in range(1, width - 1):

        sum_x = 0
        sum_y = 0
        sum_lap = 0

        for m in range(-1, 2):
            for n in range(-1, 2):

                pixel = gray[i + m, j + n]

                sum_x += pixel * sobel_x[m + 1][n + 1]
                sum_y += pixel * sobel_y[m + 1][n + 1]

                sum_lap += pixel * laplacian_kernel[m + 1][n + 1]

        gx[i, j] = sum_x
        gy[i, j] = sum_y

        sobel[i, j] = np.sqrt(
            sum_x ** 2 +
            sum_y ** 2
        )

        laplacian[i, j] = sum_lap

# ==========================================
# Print Matrices
# ==========================================
np.set_printoptions(precision=2, suppress=True)

print("\n========== Sobel ==========\n")
print(sobel)

print("\n========== Laplacian ==========\n")
print(laplacian)

# ==========================================
# Save Matrices
# ==========================================
np.savetxt("sobel_output.txt", sobel, fmt="%.2f")
np.savetxt("laplacian_output.txt", laplacian, fmt="%.2f")

# ==========================================
# Convert Negative Values
# ==========================================
laplacian = np.abs(laplacian)

# ==========================================
# Normalize
# ==========================================
sobel_display = sobel.copy()

if sobel_display.max() != 0:
    sobel_display = sobel_display / sobel_display.max() * 255

laplacian_display = laplacian.copy()

if laplacian_display.max() != 0:
    laplacian_display = laplacian_display / laplacian_display.max() * 255

# ==========================================
# Display
# ==========================================
plt.figure(figsize=(15,8))

plt.subplot(2,2,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(sobel_display.astype(np.uint8), cmap="gray")
plt.title("Sobel Edge Detection")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(laplacian_display.astype(np.uint8), cmap="gray")
plt.title("Laplacian Edge Detection")
plt.axis("off")

plt.tight_layout()
plt.show()