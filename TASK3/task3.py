import cv2
import matplotlib.pyplot as plt

# Read grayscale image
image = cv2.imread("../TASK1/image.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found!")
    exit()

# Get image dimensions
height, width = image.shape

minimum = image[0][0]
maximum = image[0][0]
total = 0

# Histogram array
histogram = [0] * 256

#loop to find minimum , maximum , and draw histogram
for row in range(height):
    for col in range(width):

        pixel = image[row][col]

        # Minimum
        if pixel < minimum:
            minimum = pixel

        # Maximum
        if pixel > maximum:
            maximum = pixel

        # Sum for average
        total += pixel

        # Histogram
        histogram[pixel] += 1

#formula for average intensity
average = total / (height * width)

#printing values
print("Pixel Intensity Analysis")
print(f"Minimum Intensity : {minimum}")
print(f"Maximum Intensity : {maximum}")
print(f"Average Intensity : {average:.2f}")


plt.figure(figsize=(10,5))
plt.bar(range(256), histogram, width=1)

plt.title("Histogram of Pixel Intensities")
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Number of Pixels")

plt.xlim(0, 255)

plt.savefig("histogram.png", dpi=300, bbox_inches="tight")

plt.show()