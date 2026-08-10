import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# TASK 3: RGB HISTOGRAM
# =========================================================


# ---------------------------------------------------------
# Function to calculate histogram manually using NumPy
# ---------------------------------------------------------

def calculate_histogram(channel):

    histogram = np.zeros(256, dtype=int)

    for pixel in channel.flatten():
        histogram[pixel] += 1

    return histogram


# ---------------------------------------------------------
# Read RGB image
# ---------------------------------------------------------

image = cv2.imread("rgb.jpg")

if image is None:
    print("Error: Image not found!")
    exit()


# ---------------------------------------------------------
# Convert BGR to RGB
# ---------------------------------------------------------

rgb_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ---------------------------------------------------------
# Separate R, G and B channels
# ---------------------------------------------------------

red_channel = rgb_image[:, :, 0]
green_channel = rgb_image[:, :, 1]
blue_channel = rgb_image[:, :, 2]


# ---------------------------------------------------------
# Calculate histograms
# ---------------------------------------------------------

red_histogram = calculate_histogram(red_channel)

green_histogram = calculate_histogram(green_channel)

blue_histogram = calculate_histogram(blue_channel)


# =========================================================
# DISPLAY IMAGE AND THREE HISTOGRAMS
# =========================================================

plt.figure(figsize=(14, 10))


# ---------------------------------------------------------
# Original RGB Image
# ---------------------------------------------------------

plt.subplot(2, 2, 1)

plt.imshow(rgb_image)

plt.title(
    "Original RGB Image",
    fontsize=14,
    fontweight="bold"
)

plt.axis("off")


# ---------------------------------------------------------
# Red Histogram
# ---------------------------------------------------------

plt.subplot(2, 2, 2)

plt.bar(
    range(256),
    red_histogram,
    width=1
)

plt.title(
    "Red Channel Histogram",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Frequency")

plt.xlim(0, 255)


# ---------------------------------------------------------
# Green Histogram
# ---------------------------------------------------------

plt.subplot(2, 2, 3)

plt.bar(
    range(256),
    green_histogram,
    width=1
)

plt.title(
    "Green Channel Histogram",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Frequency")

plt.xlim(0, 255)


# ---------------------------------------------------------
# Blue Histogram
# ---------------------------------------------------------

plt.subplot(2, 2, 4)

plt.bar(
    range(256),
    blue_histogram,
    width=1
)

plt.title(
    "Blue Channel Histogram",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Frequency")

plt.xlim(0, 255)


# ---------------------------------------------------------
# Main Title
# ---------------------------------------------------------

plt.suptitle(
    "Task 3: RGB Histogram",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()

plt.show()