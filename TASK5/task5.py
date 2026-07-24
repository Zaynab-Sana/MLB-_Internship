import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("../TASK4/noisy_image.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found!")
    exit()

height, width = image.shape


#MEAN FILTER
mean_output = np.zeros((height, width), dtype=np.uint8)

mean_kernel = [
    [1,1,1],
    [1,1,1],
    [1,1,1]
]

for row in range(1,height-1):
    for col in range(1,width-1):

        total = 0

        for i in range(-1,2):
            for j in range(-1,2):

                total += int(image[row+i][col+j] * mean_kernel[i+1][j+1])

        mean_output[row][col] = total // 9



# BOX FILTER
box_output = np.zeros((height,width),dtype=np.uint8)

box_kernel = [
    [1,1,1],
    [1,1,1],
    [1,1,1]
]

for row in range(1,height-1):
    for col in range(1,width-1):

        total = 0

        for i in range(-1,2):
            for j in range(-1,2):

                total += int(image[row+i][col+j] * box_kernel[i+1][j+1])

        box_output[row][col] = total // 9


# GAUSSIAN FILTER
gaussian_output = np.zeros((height,width),dtype=np.uint8)

gaussian_kernel = [
    [1,2,1],
    [2,4,2],
    [1,2,1]
]

for row in range(1,height-1):
    for col in range(1,width-1):

        total = 0

        for i in range(-1,2):
            for j in range(-1,2):

                total += int(int(image[row+i][col+j]) * gaussian_kernel[i+1][j+1])

        gaussian_output[row][col] = total // 16


# MEDIAN FILTER
median_output = image.copy()

for row in range(1,height-1):
    for col in range(1,width-1):

        neighbors=[]

        for i in range(-1,2):
            for j in range(-1,2):

                neighbors.append(image[row+i][col+j])

        median_output[row][col] = np.median(neighbors)


#   DISPLAY ALL RESULTS
plt.figure(figsize=(15,8))

plt.subplot(2,3,1)
plt.imshow(image,cmap="gray")
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(mean_output,cmap="gray")
plt.title("Mean Filter")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(box_output,cmap="gray")
plt.title("Box Filter")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(gaussian_output,cmap="gray")
plt.title("Gaussian Filter")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(median_output,cmap="gray")
plt.title("Median Filter")
plt.axis("off")

#COMPARISON TABLE
print("\n{:^18}|{:^12}|{:^12}|{:^12}|{:^12}".format(
    "Feature","Mean","Box","Gaussian","Median"))
print("-"*72)

print("{:<18}|{:^12}|{:^12}|{:^12}|{:^12}".format(
    "Noise Removal","Good","Good","Very Good","Excellent"))

print("{:<18}|{:^12}|{:^12}|{:^12}|{:^12}".format(
    "Edge Preserve","Poor","Poor","Good","Excellent"))

print("{:<18}|{:^12}|{:^12}|{:^12}|{:^12}".format(
    "Blur Level","High","High","Medium","Low"))

print("{:<18}|{:^12}|{:^12}|{:^12}|{:^12}".format(
    "Best Use","Random","Smoothing","Gaussian","Salt&Pepper"))

print("{:<18}|{:^12}|{:^12}|{:^12}|{:^12}".format(
    "Advantages","Simple","Fast","Better Edges","Best Noise"))

print("{:<18}|{:^12}|{:^12}|{:^12}|{:^12}".format(
    "Disadvantages","Blurs","Blurs","Slower","More Computation"))

plt.tight_layout()
plt.show()


    