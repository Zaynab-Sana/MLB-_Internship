import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read grayscale image
image = cv2.imread("../TASK6/image3.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found!")
    exit()


height, width = image.shape

#function for zero padding
def zero_padding(image, padding):

    height, width = image.shape

    padded_image = np.zeros(
        (height + 2*padding, width + 2*padding),
        dtype=np.uint8
    )


    for row in range(height):
        for col in range(width):

            padded_image[row+padding][col+padding] = image[row][col]


    return padded_image

#function for same padding
def same_padding(image, kernel_size):

    padding = kernel_size // 2

    return zero_padding(image, padding)

#applying padding
zero_padded = zero_padding(image, 1)

same_padded = same_padding(image, 3)



# Display results

plt.figure(figsize=(12,4))


plt.subplot(1,3,1)
plt.imshow(image,cmap="gray")
plt.title(f"Original\n{image.shape}")
plt.axis("off")


plt.subplot(1,3,2)
plt.imshow(zero_padded,cmap="gray")
plt.title(f"Zero Padding\n{zero_padded.shape}")
plt.axis("off")


plt.subplot(1,3,3)
plt.imshow(same_padded,cmap="gray")
plt.title(f"Same Padding\n{same_padded.shape}")
plt.axis("off")


plt.show()



print("Original Size:", image.shape)
print("Zero Padding Size:", zero_padded.shape)
print("Same Padding Size:", same_padded.shape)


print("\nExplanation:")
print("----------------------------")
print("Zero padding adds zeros around the image.")
print("Same padding adds padding so output size remains equal to input size.")
print("Padding allows convolution to process border pixels.")