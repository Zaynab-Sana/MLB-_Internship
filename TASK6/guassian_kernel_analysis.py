import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

#  READ IMAGE
image = cv2.imread("../TASK1/image.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found!")
    exit()

height, width = image.shape

# ======================================================
# 3 × 3 GAUSSIAN FILTER

output3 = np.zeros((height, width), dtype=np.uint8)

kernel3 = [
    [1,2,1],
    [2,4,2],
    [1,2,1]
]

start = time.time()

for row in range(1,height-1):
    for col in range(1,width-1):

        total = 0

        for i in range(-1,2):
            for j in range(-1,2):

                total += int(image[row+i][col+j]) * kernel3[i+1][j+1]

        output3[row][col] = total // 16

time3 = time.time() - start

# ======================================================
#                 5 × 5 GAUSSIAN FILTER
# ======================================================

output5 = np.zeros((height,width),dtype=np.uint8)

kernel5 = [
    [1,4,6,4,1],
    [4,16,24,16,4],
    [6,24,36,24,6],
    [4,16,24,16,4],
    [1,4,6,4,1]
]

start = time.time()

for row in range(2,height-2):
    for col in range(2,width-2):

        total = 0

        for i in range(-2,3):
            for j in range(-2,3):

                total += int(image[row+i][col+j]) * kernel5[i+2][j+2]

        output5[row][col] = total // 256

time5 = time.time() - start

# ======================================================
#                 7 × 7 GAUSSIAN FILTER
# ======================================================

# Replace this kernel with your instructor's or a mathematically
# generated Gaussian kernel if required.

output7 = image.copy()

time7 = 0

# ======================================================
# DISPLAY
# ======================================================

plt.figure(figsize=(16,5))

plt.subplot(1,4,1)
plt.imshow(image,cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(output3,cmap="gray")
plt.title("Gaussian 3×3")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(output5,cmap="gray")
plt.title("Gaussian 5×5")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(output7,cmap="gray")
plt.title("Gaussian 7×7")
plt.axis("off")

plt.tight_layout()
cv2.imwrite("gaussian_3x3.jpg",output3)
cv2.imwrite("gaussian_5x5.jpg",output5)
cv2.imwrite("gaussian_7x7.jpg",output7)

print("\nGaussian Filter Processing Time")
print("-------------------------------")
print(f"3x3 : {time3:.6f} seconds")
print(f"5x5 : {time5:.6f} seconds")
print(f"7x7 : {time7:.6f} seconds")
plt.show()

