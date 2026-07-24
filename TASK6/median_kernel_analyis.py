import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# ===================== READ IMAGE =====================

image = cv2.imread("../TASK1/image.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found!")
    exit()

height, width = image.shape

# ======================================================
#                 3 × 3 MEDIAN FILTER
# ======================================================

output3 = image.copy()

start = time.time()

for row in range(1, height-1):
    for col in range(1, width-1):

        neighbors = []

        for i in range(-1,2):
            for j in range(-1,2):
                neighbors.append(image[row+i][col+j])

        output3[row][col] = np.median(neighbors)

time3 = time.time() - start

# ======================================================
#                 5 × 5 MEDIAN FILTER
# ======================================================

output5 = image.copy()

start = time.time()

for row in range(2, height-2):
    for col in range(2, width-2):

        neighbors = []

        for i in range(-2,3):
            for j in range(-2,3):
                neighbors.append(image[row+i][col+j])

        output5[row][col] = np.median(neighbors)

time5 = time.time() - start

# ======================================================
#                 7 × 7 MEDIAN FILTER
# ======================================================

output7 = image.copy()

start = time.time()

for row in range(3, height-3):
    for col in range(3, width-3):

        neighbors = []

        for i in range(-3,4):
            for j in range(-3,4):
                neighbors.append(image[row+i][col+j])

        output7[row][col] = np.median(neighbors)

time7 = time.time() - start

# ======================================================
#                    DISPLAY RESULTS
# ======================================================

plt.figure(figsize=(16,5))

plt.subplot(1,4,1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(output3, cmap="gray")
plt.title("Median 3×3")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(output5, cmap="gray")
plt.title("Median 5×5")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(output7, cmap="gray")
plt.title("Median 7×7")
plt.axis("off")

plt.tight_layout()

# ======================================================
# SAVE IMAGES
# ======================================================

cv2.imwrite("median_3x3.jpg", output3)
cv2.imwrite("median_5x5.jpg", output5)
cv2.imwrite("median_7x7.jpg", output7)

# ======================================================
# PROCESSING TIME
# ======================================================

print("\nMedian Filter Processing Time")
print("-----------------------------")
print(f"3×3 : {time3:.6f} seconds")
print(f"5×5 : {time5:.6f} seconds")
print(f"7×7 : {time7:.6f} seconds")

# ======================================================
# OBSERVATION TABLE
# ======================================================

print("\nKernel Size Analysis")
print("-"*75)
print("{:<12}{:<20}{:<20}{:<15}".format(
    "Kernel", "Sharpness", "Noise Removal", "Blur"))

print("-"*75)

print("{:<12}{:<20}{:<20}{:<15}".format(
    "3×3", "High", "Good", "Low"))

print("{:<12}{:<20}{:<20}{:<15}".format(
    "5×5", "Medium", "Very Good", "Medium"))

print("{:<12}{:<20}{:<20}{:<15}".format(
    "7×7", "Low", "Excellent", "High"))
plt.show()

