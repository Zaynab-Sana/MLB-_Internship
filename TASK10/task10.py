import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("../TASK6/image3.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
    exit()


print("Original Image Size:", image.shape)

#1-zero padding function
def zero_padding(image, padding_size):

    height, width = image.shape

    new_height = height + 2 * padding_size
    new_width = width + 2 * padding_size

    padded_image = np.zeros((new_height, new_width), dtype=np.uint8)


    # copy original image into center

    for i in range(height):
        for j in range(width):

            padded_image[i + padding_size][j + padding_size] = image[i][j]


    return padded_image

#2-function for convulution
def convolution(image, kernel, stride):

    image_height, image_width = image.shape
    kernel_size = kernel.shape[0]

    output_height = ((image_height - kernel_size)//stride) + 1
    output_width = ((image_width - kernel_size)//stride) + 1

    output = np.zeros((output_height, output_width))


    for i in range(0, output_height):

        for j in range(0, output_width):

            sum_value = 0

            for m in range(kernel_size):

                for n in range(kernel_size):

                    image_value = image[i*stride + m][j*stride + n]

                    kernel_value = kernel[m][n]

                    sum_value += image_value * kernel_value



            output[i][j] = sum_value



    # normalize values between 0-255

    output = np.clip(output,0,255)

    return output.astype(np.uint8)





# 3. Select Kernel
# Edge Detection Kernel

kernel = np.array([
    [-1,-1,-1],
    [-1, 8,-1],
    [-1,-1,-1]
])

#applying padding
padded_image = zero_padding(image,1)
print("Padded Image Size:", padded_image.shape)



#stride 1
output_stride1 = convolution(
    padded_image,
    kernel,
    stride=1
)
print("Output Size with Stride 1:",
      output_stride1.shape)

#stride 2
output_stride2 = convolution(
    padded_image,
    kernel,
    stride=2
)

print("Output Size with Stride 2:",
      output_stride2.shape)

#displaying results
plt.figure(figsize=(12,8))


plt.subplot(2,2,1)
plt.imshow(image,cmap="gray")
plt.title("Original Image")
plt.axis("off")


plt.subplot(2,2,2)
plt.imshow(padded_image,cmap="gray")
plt.title("Zero Padded Image")
plt.axis("off")


plt.subplot(2,2,3)
plt.imshow(output_stride1,cmap="gray")
plt.title("Convolution Stride = 1")
plt.axis("off")


plt.subplot(2,2,4)
plt.imshow(output_stride2,cmap="gray")
plt.title("Convolution Stride = 2")
plt.axis("off")


plt.show()