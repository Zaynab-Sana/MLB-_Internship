import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read Image
image = cv2.imread("../TASK1/greyscale_image.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found!")
    exit()

height, width = image.shape


# Motion Blur Function
def motion_blur(img, kernel_size):

    # Create Motion Blur Kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    # Middle row becomes 1
    kernel[kernel_size // 2] = 1

    # Normalize kernel
    kernel = kernel / kernel_size

    pad = kernel_size // 2

    output = img.copy()

    for i in range(pad, height - pad):
        for j in range(pad, width - pad):

            region = img[i-pad:i+pad+1, j-pad:j+pad+1]

            value = np.sum(region * kernel)

            if value < 0:
                value = 0
            elif value > 255:
                value = 255

            output[i, j] = int(value)

    return output


# Apply Motion Blur
blur3 = motion_blur(image, 3)
blur5 = motion_blur(image, 5)
blur7 = motion_blur(image, 7)


# Display Results
plt.figure(figsize=(16,5))

plt.subplot(1,4,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(blur3, cmap='gray')
plt.title("Motion Blur (3×3 Kernel)")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(blur5, cmap='gray')
plt.title("Motion Blur (5×5 Kernel)")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(blur7, cmap='gray')
plt.title("Motion Blur (7×7 Kernel)")
plt.axis("off")

plt.tight_layout()
print("\n========== Motion Blur ==========\n")

print("3×3 Kernel : Slight horizontal motion effect.")
print("5×5 Kernel : Moderate motion blur.")
print("7×7 Kernel : Strong motion blur with more loss of details.")

print("\nCommon Uses:")
print("- Simulates camera or object movement.")
print("- Used in photography and computer graphics.")
print("- Helps study motion blur and image restoration techniques.")
plt.show()

