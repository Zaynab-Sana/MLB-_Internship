import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Read Image
# -------------------------------------------------
image = cv2.imread("../TASK1/image.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found!")
    exit()

# Convert image to float for calculations
image = image.astype(np.float32)

height, width = image.shape

# -------------------------------------------------
# Sobel Kernels
# -------------------------------------------------
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

# -------------------------------------------------
# Output Images
# -------------------------------------------------
gradient_x = np.zeros((height, width), dtype=np.float32)
gradient_y = np.zeros((height, width), dtype=np.float32)

# -------------------------------------------------
# Manual Convolution
# -------------------------------------------------
for i in range(1, height - 1):
    for j in range(1, width - 1):

        gx = 0
        gy = 0

        for m in range(-1, 2):
            for n in range(-1, 2):

                pixel = image[i + m, j + n]

                gx += pixel * sobel_x[m + 1, n + 1]
                gy += pixel * sobel_y[m + 1, n + 1]

        gradient_x[i, j] = gx
        gradient_y[i, j] = gy

# -------------------------------------------------
# Gradient Magnitude
# -------------------------------------------------
gradient_magnitude = np.sqrt(
    gradient_x ** 2 + gradient_y ** 2
)

# -------------------------------------------------
# Gradient Direction (Angle)
# -------------------------------------------------
gradient_direction = np.arctan2(
    gradient_y,
    gradient_x
)

gradient_direction = np.degrees(gradient_direction)

# -------------------------------------------------
# Normalize Images for Display
# -------------------------------------------------
gx_display = cv2.normalize(
    np.abs(gradient_x),
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

gy_display = cv2.normalize(
    np.abs(gradient_y),
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

magnitude_display = cv2.normalize(
    gradient_magnitude,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

direction_display = cv2.normalize(
    gradient_direction,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

gx_display = gx_display.astype(np.uint8)
gy_display = gy_display.astype(np.uint8)
magnitude_display = magnitude_display.astype(np.uint8)
direction_display = direction_display.astype(np.uint8)

# -------------------------------------------------
# Display Results
# -------------------------------------------------
plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(gx_display, cmap="gray")
plt.title("Gradient X")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(gy_display, cmap="gray")
plt.title("Gradient Y")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(magnitude_display, cmap="gray")
plt.title("Gradient Magnitude")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(direction_display, cmap="gray")
plt.title("Gradient Direction")
plt.axis("off")

plt.tight_layout()
# -------------------------------------------------
# Print Sample Values
# -------------------------------------------------
center_x = height // 2
center_y = width // 2

print("Gradient X at center:", gradient_x[center_x, center_y])
print("Gradient Y at center:", gradient_y[center_x, center_y])
print("Gradient Magnitude:", gradient_magnitude[center_x, center_y])
print("Gradient Direction:", gradient_direction[center_x, center_y], "degrees")
plt.show()

