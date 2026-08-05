import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Read Image
# ==========================================
image = cv2.imread("image.jpg")

if image is None:
    print("Image not found!")
    exit()

# Convert BGR to RGB (Only for Display)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

height, width, channels = image.shape

# ==========================================
# Convert RGB to Grayscale Manually
# Gray = 0.299R + 0.587G + 0.114B
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
# Create Empty Arrays
# ==========================================
gx = np.zeros((height, width), dtype=np.float32)
gy = np.zeros((height, width), dtype=np.float32)

gradient_magnitude = np.zeros((height, width), dtype=np.float32)
gradient_direction = np.zeros((height, width), dtype=np.float32)

# ==========================================
# Manual Gradient Calculation
# ==========================================
for i in range(1, height - 1):
    for j in range(1, width - 1):

        # Horizontal Gradient
        gx[i, j] = gray[i, j + 1] - gray[i, j - 1]

        # Vertical Gradient
        gy[i, j] = gray[i + 1, j] - gray[i - 1, j]

        # Gradient Magnitude
        gradient_magnitude[i, j] = np.sqrt(
            gx[i, j] ** 2 +
            gy[i, j] ** 2
        )

        # Gradient Direction (Degrees)
        gradient_direction[i, j] = np.degrees(
            np.arctan2(gy[i, j], gx[i, j])
        )

# Convert negative angles to 0–180
gradient_direction[gradient_direction < 0] += 180

# ==========================================
# Print Computed Values
# ==========================================
np.set_printoptions(precision=2, suppress=True)

print("\n==================== Gx ====================\n")
print(gx)

print("\n==================== Gy ====================\n")
print(gy)

print("\n============= Gradient Magnitude ============\n")
print(gradient_magnitude)

print("\n============= Gradient Direction ============\n")
print(gradient_direction)

# ==========================================
# Save Results to Text Files
# ==========================================
np.savetxt("gx.txt", gx, fmt="%.2f")
np.savetxt("gy.txt", gy, fmt="%.2f")
np.savetxt("gradient_magnitude.txt", gradient_magnitude, fmt="%.2f")
np.savetxt("gradient_direction.txt", gradient_direction, fmt="%.2f")

print("\nGradient matrices saved successfully.")

# ==========================================
# Normalize Images for Display
# ==========================================
gx_display = np.abs(gx)

if gx_display.max() != 0:
    gx_display = gx_display / gx_display.max() * 255

gy_display = np.abs(gy)

if gy_display.max() != 0:
    gy_display = gy_display / gy_display.max() * 255

magnitude_display = gradient_magnitude

if magnitude_display.max() != 0:
    magnitude_display = magnitude_display / magnitude_display.max() * 255

direction_display = gradient_direction

if direction_display.max() != 0:
    direction_display = direction_display / direction_display.max() * 255

# ==========================================
# Display Results
# ==========================================
plt.figure(figsize=(15,8))

plt.subplot(2,3,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(gx_display.astype(np.uint8), cmap="gray")
plt.title("Gradient X (Gx)")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(gy_display.astype(np.uint8), cmap="gray")
plt.title("Gradient Y (Gy)")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(magnitude_display.astype(np.uint8), cmap="gray")
plt.title("Gradient Magnitude")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(direction_display.astype(np.uint8), cmap="hsv")
plt.title("Gradient Direction")
plt.axis("off")

plt.tight_layout()
plt.show()