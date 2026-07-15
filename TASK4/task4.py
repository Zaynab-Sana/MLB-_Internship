import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image by open cv
image = cv2.imread("image2.jpg")

#Validation if image is loaded or not
if image is None:
    print("Image not found!")
    exit()

# Get image dimensions
rows=image.shape[0]
cols=image.shape[1]

# Create empty HSV image
hsv_image = np.zeros((rows, cols, 3), dtype=np.uint8)


# MANUAL conversion of BGR TO HSV CONVERSION
for i in range(rows):
    for j in range(cols):

        # Normalize
        b = image[i][j][0] / 255.0
        g = image[i][j][1] / 255.0
        r = image[i][j][2] / 255.0

        cmax = max(r, g, b)
        cmin = min(r, g, b)
        delta = cmax - cmin

        # Calculate Hue
        if delta == 0:
            h = 0

        elif cmax == r:
            h = 60 * (((g - b) / delta) % 6)

        elif cmax == g:
            h = 60 * (((b - r) / delta) + 2)

        else:
            h = 60 * (((r - g) / delta) + 4)

        # Calculate Saturation
        if cmax == 0:
            s = 0
        else:
            s = delta / cmax

        # Calculate Value
        v = cmax

        # Store HSV values
        hsv_image[i][j][0] = int(h / 2)      # Hue (0-179)
        hsv_image[i][j][1] = int(s * 255)    # Saturation (0-255)
        hsv_image[i][j][2] = int(v * 255)    # Value (0-255)

#Detection of blue color

lower_h = 100
lower_s = 150
lower_v = 50

upper_h = 140
upper_s = 255
upper_v = 255

mask = np.zeros((rows, cols), dtype=np.uint8)

for i in range(rows):
    for j in range(cols):

        h = hsv_image[i][j][0]
        s = hsv_image[i][j][1]
        v = hsv_image[i][j][2]

        if (lower_h <= h <= upper_h and
            lower_s <= s <= upper_s and
            lower_v <= v <= upper_v):

            mask[i][j] = 255

        else:
            mask[i][j] = 0

# MANUAL IMAGE SEGMENTATION

segmented_image = np.zeros_like(image)

for i in range(rows):
    for j in range(cols):

        if mask[i][j] == 255:
            segmented_image[i][j] = image[i][j]
        else:
            segmented_image[i][j] = [0, 0, 0]

# MANUAL BGR TO RGB FOR DISPLAY

rgb_original = image.copy()
rgb_segmented = segmented_image.copy()

for i in range(rows):
    for j in range(cols):

        # Original
        blue = rgb_original[i][j][0]
        red = rgb_original[i][j][2]

        rgb_original[i][j][0] = red
        rgb_original[i][j][2] = blue

        # Segmented
        blue = rgb_segmented[i][j][0]
        red = rgb_segmented[i][j][2]

        rgb_segmented[i][j][0] = red
        rgb_segmented[i][j][2] = blue

# DISPLAY AND SAVE

plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
plt.imshow(rgb_original)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(rgb_segmented)
plt.title("Segmented Image")
plt.axis("off")

plt.tight_layout()

plt.savefig("task4_output.png", dpi=300, bbox_inches="tight")

print("Task 4 completed successfully!")