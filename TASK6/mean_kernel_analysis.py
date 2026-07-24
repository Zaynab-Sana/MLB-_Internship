import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

#read image
image = cv2.imread("../TASK1/image.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found!")
    exit()

height, width = image.shape

# 3 × 3 MEAN FILTER
output3 = np.zeros((height, width), dtype=np.uint8)

start = time.time()

kernel3 = [
    [1,1,1],
    [1,1,1],
    [1,1,1]
]

for row in range(1, height-1):
    for col in range(1, width-1):

        total = 0

        for i in range(-1,2):
            for j in range(-1,2):

                pixel =int( image[row+i][col+j])
                kernel_value = kernel3[i+1][j+1]

                total += pixel * kernel_value

        output3[row][col] = total // 9

time3 = time.time() - start

#  5 × 5 MEAN FILTER
output5 = np.zeros((height, width), dtype=np.uint8)

start = time.time()

kernel5 = [[1]*5 for _ in range(5)]

for row in range(2, height-2):
    for col in range(2, width-2):

        total = 0

        for i in range(-2,3):
            for j in range(-2,3):

                pixel = int(image[row+i][col+j])
                kernel_value = kernel5[i+2][j+2]

                total += pixel * kernel_value

        output5[row][col] = total // 25

time5 = time.time() - start

# 7 × 7 MEAN FILTER
output7 = np.zeros((height, width), dtype=np.uint8)

start = time.time()

kernel7 = [[1]*7 for _ in range(7)]

for row in range(3, height-3):
    for col in range(3, width-3):

        total = 0

        for i in range(-3,4):
            for j in range(-3,4):

                pixel = int(image[row+i][col+j])
                kernel_value = kernel7[i+3][j+3]

                total += pixel * kernel_value

        output7[row][col] = total // 49

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
plt.title("Mean 3×3")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(output5, cmap="gray")
plt.title("Mean 5×5")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(output7, cmap="gray")
plt.title("Mean 7×7")
plt.axis("off")

plt.tight_layout()

cv2.imwrite("mean_3x3.jpg", output3)
cv2.imwrite("mean_5x5.jpg", output5)
cv2.imwrite("mean_7x7.jpg", output7)

# PROCESSING TIME
print("\nMean Filter Processing Time")
print("---------------------------")
print(f"3x3 : {time3:.6f} seconds")
print(f"5x5 : {time5:.6f} seconds")
print(f"7x7 : {time7:.6f} seconds")


print("\nKernel Size Analysis")
print("-"*75)
print("{:<12}{:<20}{:<20}{:<15}".format(
    "Kernel","Sharpness","Noise Removal","Blur"))

print("-"*75)

print("{:<12}{:<20}{:<20}{:<15}".format(
    "3x3","High","Low","Low"))

print("{:<12}{:<20}{:<20}{:<15}".format(
    "5x5","Medium","Good","Medium"))

print("{:<12}{:<20}{:<20}{:<15}".format(
    "7x7","Low","Excellent","High"))
plt.show()

