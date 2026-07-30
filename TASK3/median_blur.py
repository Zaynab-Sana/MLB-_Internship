import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read Image
image = cv2.imread("../TASK1/greyscale_image.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found!")
    exit()

height, width = image.shape


# Median Blur Function
def median_blur(img, kernel_size):

    pad = kernel_size // 2

    output = img.copy()

    for i in range(pad, height - pad):
        for j in range(pad, width - pad):

            # Extract neighborhood
            region = img[i-pad:i+pad+1, j-pad:j+pad+1]

            # Convert to 1D array
            pixels = region.flatten()

            # Sort pixel values
            pixels.sort()

            # Find median
            median = pixels[len(pixels) // 2]

            output[i, j] = median

    return output


# Apply Median Blur
blur3 = median_blur(image, 3)
blur5 = median_blur(image, 5)
blur7 = median_blur(image, 7)


# Display Results
plt.figure(figsize=(16,5))

# Original Image
plt.subplot(1,4,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Median Blur 3x3
plt.subplot(1,4,2)
plt.imshow(blur3, cmap='gray')
plt.title("Median Blur (3×3 Kernel)")
plt.axis("off")

# Median Blur 5x5
plt.subplot(1,4,3)
plt.imshow(blur5, cmap='gray')
plt.title("Median Blur (5×5 Kernel)")
plt.axis("off")

# Median Blur 7x7
plt.subplot(1,4,4)
plt.imshow(blur7, cmap='gray')
plt.title("Median Blur (7×7 Kernel)")
plt.axis("off")

plt.tight_layout()

print("\n========== Median Blur ==========\n")

print("3×3 Kernel : Slight smoothing while preserving image details.")
print("5×5 Kernel : Removes more noise with moderate blurring.")
print("7×7 Kernel : Strong smoothing but reduces fine details.")

print("\nCommon Uses:")
print("- Removes Salt-and-Pepper Noise.")
print("- Preserves edges better than Mean Filter.")
print("- Used in image preprocessing before segmentation and edge detection.")
plt.show()


