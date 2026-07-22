import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread("../TASK6/image3.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found!")
    exit()

height, width = image.shape

#1-Identity kernel
identity_kernel = [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
]


#2-blur kernel
blur_kernel = [
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
]

#3-sharpen kernel
sharpen_kernel = [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
]

#4-edge kernel
edge_kernel = [
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
]

#function for manual convulation
def manual_convolution(image, kernel):

    height, width = image.shape

    output = np.zeros((height, width), dtype=np.uint8)

     #outer loop to move kernel around centre
    for row in range(1, height - 1):
        for col in range(1, width - 1):

            total = 0

            #inner loop to access 9 pixels
            for i in range(-1, 2):
                for j in range(-1, 2):

                    #Formula to find pixel and kernel value
                    pixel =int( image[row + i][col + j])
                    kernel_value = kernel[i + 1][j + 1]

                    #output value
                    total += pixel * kernel_value

            if total < 0:
                total = 0

            if total > 255:
                total = 255

            output[row][col] = int(total)

    return output



identity_output = manual_convolution(image, identity_kernel)
blur_output = manual_convolution(image, blur_kernel)
sharpen_output = manual_convolution(image, sharpen_kernel)
edge_output = manual_convolution(image, edge_kernel)

#saving images
cv2.imwrite("identity_output.jpg", identity_output)
cv2.imwrite("blur_output.jpg", blur_output)
cv2.imwrite("sharpen_output.jpg", sharpen_output)
cv2.imwrite("edge_output.jpg", edge_output)

#displaying results
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(identity_output, cmap="gray")
plt.title("Identity")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(blur_output, cmap="gray")
plt.title("Blur")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(sharpen_output, cmap="gray")
plt.title("Sharpen")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(edge_output, cmap="gray")
plt.title("Edge Detection")
plt.axis("off")

plt.tight_layout()
plt.show()

print("\nKernel Comparison")
print("---------------------------------------------")

print("1. Identity Kernel")
print("- Keeps the image almost the same.")
print("- Only the center pixel is copied to the output.")

print("\n2. Blur Kernel")
print("- Averages neighboring pixels.")
print("- Produces a smooth and blurred image.")
print("- Reduces noise and fine details.")

print("\n3. Sharpen Kernel")
print("- Increases the intensity of the center pixel.")
print("- Makes edges and details clearer.")
print("- The image appears sharper.")

print("\n4. Edge Detection Kernel")
print("- Detects sudden changes in pixel intensity.")
print("- Highlights object boundaries and edges.")
print("- Flat regions become dark while edges become bright.")