import cv2
import numpy as np
import matplotlib.pyplot as plt


# ===================== READ IMAGES =====================

dark_image = cv2.imread("dark_image.jpg")
bright_image = cv2.imread("bright_image.jpeg")
normal_image = cv2.imread("normal_image.jpg")


if dark_image is None or bright_image is None or normal_image is None:
    print("Image Not Found")
    exit()


# Function to convert image into grayscale manually
def convert_to_grayscale(image):

    num_rows = image.shape[0]
    num_col = image.shape[1]

    grayscale_image = np.zeros((num_rows, num_col), dtype=np.uint8)

    for i in range(num_rows):
        for j in range(num_col):

            b = image[i, j, 0]
            g = image[i, j, 1]
            r = image[i, j, 2]

            grey = (0.299*r) + (0.587*g) + (0.114*b)

            grayscale_image[i, j] = grey

    return grayscale_image



# Function to calculate histogram manually
def calculate_histogram(grayscale_image):

    histogram = np.zeros(256, dtype=int)

    num_rows = grayscale_image.shape[0]
    num_col = grayscale_image.shape[1]

    for i in range(num_rows):
        for j in range(num_col):

            pixel_value = grayscale_image[i, j]

            histogram[pixel_value] += 1

    return histogram



# ===================== DARK IMAGE =====================

dark_gray = convert_to_grayscale(dark_image)

dark_histogram = calculate_histogram(dark_gray)



# ===================== BRIGHT IMAGE =====================

bright_gray = convert_to_grayscale(bright_image)

bright_histogram = calculate_histogram(bright_gray)



# ===================== NORMAL IMAGE =====================

normal_gray = convert_to_grayscale(normal_image)

normal_histogram = calculate_histogram(normal_gray)



# ===================== PLOT HISTOGRAMS SIDE BY SIDE =====================

intensity = np.arange(256)


plt.figure(figsize=(15,5))


# Dark Histogram
plt.subplot(1,3,1)
plt.plot(intensity, dark_histogram)
plt.title("Dark Image Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xlim([0,255])


# Bright Histogram
plt.subplot(1,3,2)
plt.plot(intensity, bright_histogram)
plt.title("Bright Image Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xlim([0,255])


# Normal Histogram
plt.subplot(1,3,3)
plt.plot(intensity, normal_histogram)
plt.title("Normal Image Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xlim([0,255])


plt.tight_layout()


# Save graph
plt.savefig("histogram_comparison.png")


# Close plot
plt.close()