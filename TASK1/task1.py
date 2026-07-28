import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
image = cv2.imread("image.jpg")

if image is None:
    print("Image not found!")
    exit()

# Get image dimensions
height, width, channels = image.shape

# Brightness value
brightness_value = 90

# Create output images
bright_image = np.zeros((height, width, channels), dtype=np.uint8)
dark_image = np.zeros((height, width, channels), dtype=np.uint8)


# Create Brighter Image
for i in range(height):
    for j in range(width):
        for k in range(channels):

            new_pixel = int(image[i, j, k]) + brightness_value

            if new_pixel > 255:
                new_pixel = 255

            bright_image[i, j, k] = new_pixel


# Create Darker Image
for i in range(height):
    for j in range(width):
        for k in range(channels):

            new_pixel = int(image[i, j, k]) - brightness_value

            if new_pixel < 0:
                new_pixel = 0

            dark_image[i, j, k] = new_pixel

# Convert BGR to RGB for Matplotlib
original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
bright_rgb = cv2.cvtColor(bright_image, cv2.COLOR_BGR2RGB)
dark_rgb = cv2.cvtColor(dark_image, cv2.COLOR_BGR2RGB)

# Display images
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(bright_rgb)
plt.title("Brighter Image")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(dark_rgb)
plt.title("Darker Image")
plt.axis("off")

plt.tight_layout()
plt.show()