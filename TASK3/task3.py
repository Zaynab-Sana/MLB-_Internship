import cv2
import numpy as np
import matplotlib.pyplot as plt


# ================= READ IMAGE =================

image = cv2.imread("low_contrast_image.jpg")

if image is None:
    print("Image Not Found")
    exit()


# ================= GET DIMENSIONS =================

num_rows = image.shape[0]
num_col = image.shape[1]


# ================= CONVERT TO GRAYSCALE =================

grayscale_image = np.zeros((num_rows, num_col), dtype=np.uint8)


for i in range(num_rows):
    for j in range(num_col):

        b = image[i,j,0]
        g = image[i,j,1]
        r = image[i,j,2]

        grey = (0.299*r) + (0.587*g) + (0.114*b)

        grayscale_image[i,j] = grey



# ================= CALCULATE HISTOGRAM =================

histogram = np.zeros(256, dtype=int)


for i in range(num_rows):
    for j in range(num_col):

        pixel = grayscale_image[i,j]

        histogram[pixel] += 1



# ================= CALCULATE CDF =================

cdf = np.zeros(256, dtype=int)

cdf[0] = histogram[0]


for i in range(1,256):

    cdf[i] = cdf[i-1] + histogram[i]



# ================= FIND CDF MIN =================

cdf_min = 0

for i in range(256):

    if cdf[i] != 0:
        cdf_min = cdf[i]
        break



# ================= NORMALIZE CDF =================

total_pixels = num_rows * num_col


normalized_cdf = np.zeros(256, dtype=np.uint8)


for i in range(256):

    normalized_cdf[i] = ((cdf[i]-cdf_min) /
                         (total_pixels-cdf_min)) * 255



# ================= CREATE EQUALIZED IMAGE =================

equalized_image = np.zeros((num_rows,num_col), dtype=np.uint8)


for i in range(num_rows):
    for j in range(num_col):

        old_pixel = grayscale_image[i,j]

        new_pixel = normalized_cdf[old_pixel]

        equalized_image[i,j] = new_pixel



# ================= SAVE IMAGES =================

cv2.imwrite("original_grayscale.png", grayscale_image)

cv2.imwrite("equalized_image.png", equalized_image)



# ================= CALCULATE EQUALIZED HISTOGRAM =================

equalized_histogram = np.zeros(256, dtype=int)


for i in range(num_rows):
    for j in range(num_col):

        pixel = equalized_image[i,j]

        equalized_histogram[pixel] += 1



# ================= PLOT COMPARISON =================

intensity = np.arange(256)


plt.figure(figsize=(12,5))


# Original Histogram
plt.subplot(1,2,1)

plt.plot(intensity, histogram)

plt.title("Original Histogram")

plt.xlabel("Pixel Intensity")

plt.ylabel("Frequency")

plt.xlim([0,255])



# Equalized Histogram
plt.subplot(1,2,2)

plt.plot(intensity, equalized_histogram)

plt.title("Equalized Histogram")

plt.xlabel("Pixel Intensity")

plt.ylabel("Frequency")

plt.xlim([0,255])


plt.tight_layout()


plt.savefig("histogram_equalization_comparison.png")


plt.close()