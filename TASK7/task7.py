
import time
import cv2
import numpy as np

# Read image
image = cv2.imread("../TASK1/image.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found!")
    exit()

height, width = image.shape

# =======================
# MEAN FILTER
# =======================

output = np.zeros((height,width),dtype=np.uint8)

start = time.perf_counter()

kernel = [
    [1,1,1],
    [1,1,1],
    [1,1,1]
]

for row in range(1,height-1):
    for col in range(1,width-1):

        total = 0

        for i in range(-1,2):
            for j in range(-1,2):

                total += int(image[row+i][col+j])

        output[row][col] = total//9

mean_time = time.perf_counter()-start


# =======================
# BOX FILTER
# =======================

output = np.zeros((height,width),dtype=np.uint8)

start = time.perf_counter()

kernel = [
    [1,1,1],
    [1,1,1],
    [1,1,1]
]

for row in range(1,height-1):
    for col in range(1,width-1):

        total = 0

        for i in range(-1,2):
            for j in range(-1,2):

                total += int(image[row+i][col+j])

        output[row][col] = total//9

box_time = time.perf_counter()-start


# =======================
# GAUSSIAN FILTER
# =======================

output = np.zeros((height,width),dtype=np.uint8)

kernel = [
    [1,2,1],
    [2,4,2],
    [1,2,1]
]

start = time.perf_counter()

for row in range(1,height-1):
    for col in range(1,width-1):

        total = 0

        for i in range(-1,2):
            for j in range(-1,2):

                total += int(image[row+i][col+j])*kernel[i+1][j+1]

        output[row][col]=total//16

gaussian_time=time.perf_counter()-start


# =======================
# MEDIAN FILTER
# =======================

output=image.copy()

start=time.perf_counter()

for row in range(1,height-1):
    for col in range(1,width-1):

        neighbors=[]

        for i in range(-1,2):
            for j in range(-1,2):

                neighbors.append(image[row+i][col+j])

        output[row][col]=np.median(neighbors)

median_time=time.perf_counter()-start


# =======================
# PRINT TABLE
# =======================

print("\nPerformance Analysis")
print("-"*55)

print("{:<15}{:<20}".format("Filter","Execution Time (s)"))

print("-"*55)

print("{:<15}{:<20.6f}".format("Mean",mean_time))
print("{:<15}{:<20.6f}".format("Box",box_time))
print("{:<15}{:<20.6f}".format("Gaussian",gaussian_time))
print("{:<15}{:<20.6f}".format("Median",median_time))