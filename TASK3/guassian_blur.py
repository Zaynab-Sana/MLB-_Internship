import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read Image
image = cv2.imread("../TASK1/greyscale_image.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found!")
    exit()

height, width = image.shape


# Gaussian Kernels
kernel3 = np.array([
    [1,2,1],
    [2,4,2],
    [1,2,1]
],dtype=np.float32)/16

kernel5 = np.array([
    [1,4,6,4,1],
    [4,16,24,16,4],
    [6,24,36,24,6],
    [4,16,24,16,4],
    [1,4,6,4,1]
],dtype=np.float32)/256

kernel7 = np.array([
    [0,0,1,2,1,0,0],
    [0,3,13,22,13,3,0],
    [1,13,59,97,59,13,1],
    [2,22,97,159,97,22,2],
    [1,13,59,97,59,13,1],
    [0,3,13,22,13,3,0],
    [0,0,1,2,1,0,0]
],dtype=np.float32)

kernel7 = kernel7 / np.sum(kernel7)

# Function
def gaussian_blur(img,kernel):

    k = kernel.shape[0]
    pad = k//2

    output = img.copy()

    for i in range(pad,height-pad):
        for j in range(pad,width-pad):

            region = img[i-pad:i+pad+1,j-pad:j+pad+1]

            value = np.sum(region*kernel)

            output[i,j] = int(value)

    return output


# Results
blur3 = gaussian_blur(image,kernel3)
blur5 = gaussian_blur(image,kernel5)
blur7 = gaussian_blur(image,kernel7)


# Display Results
plt.figure(figsize=(16,5))

# Original Image
plt.subplot(1,4,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Gaussian Blur 3x3
plt.subplot(1,4,2)
plt.imshow(blur3, cmap='gray')
plt.title("Gaussian Blur (3×3 Kernel)")
plt.axis("off")

# Gaussian Blur 5x5
plt.subplot(1,4,3)
plt.imshow(blur5, cmap='gray')
plt.title("Gaussian Blur (5×5 Kernel)")
plt.axis("off")

# Gaussian Blur 7x7
plt.subplot(1,4,4)
plt.imshow(blur7, cmap='gray')
plt.title("Gaussian Blur (7×7 Kernel)")
plt.axis("off")

plt.tight_layout()


print("\nGaussian Blur")
print("3x3 -> Slight blur")
print("5x5 -> Moderate blur")
print("7x7 -> Strong blur")
print("Common Use: Noise removal before edge detection.")

plt.show()
