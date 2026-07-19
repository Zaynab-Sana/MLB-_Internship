import cv2
import numpy as np

image = cv2.imread("../reading_image/image.jpg")

if image is None:
    print("Image Not Found")
    exit()

# Get image dimensions
num_rows = image.shape[0]
num_cols = image.shape[1]
channels = image.shape[2]

# Ask user for flip option
while True:
    print("Select Flip Option:")
    print("1. Horizontal Flip")
    print("2. Vertical Flip")
    print("3. Horizontal + Vertical Flip")

    choice = int(input("Enter your choice: "))

    if choice in [1, 2, 3]:
        break
    else:
        print("Invalid Choice! Try Again.\n")

# Create an empty image
flipped_image = np.zeros((num_rows, num_cols, channels), dtype=np.uint8)

# Horizontal Flip
if choice == 1:
    for i in range(num_rows):
        for j in range(num_cols):
            flipped_image[i, num_cols - 1 - j] = image[i, j]

# Vertical Flip
elif choice == 2:
    for i in range(num_rows):
        for j in range(num_cols):
            flipped_image[num_rows - 1 - i, j] = image[i, j]

# Horizontal + Vertical Flip
elif choice == 3:
    for i in range(num_rows):
        for j in range(num_cols):
            flipped_image[num_rows - 1 - i, num_cols - 1 - j] = image[i, j]

# Save image
saved = cv2.imwrite("flipped_image.png", flipped_image)

if saved:
    print("Image Saved Successfully")
else:
    print("Image Not Saved")

