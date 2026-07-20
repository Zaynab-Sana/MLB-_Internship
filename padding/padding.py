import cv2
import numpy as np

image = cv2.imread("../reading_image/image.jpg")

if image is None:
    print("Image not found")
    exit()

height, width, channels = image.shape

# User input
while True:
    try:
        top = int(input("Enter top padding: "))
        bottom = int(input("Enter bottom padding: "))
        left = int(input("Enter left padding: "))
        right = int(input("Enter right padding: "))

        if top < 0 or bottom < 0 or left < 0 or right < 0:
            print("Padding values cannot be negative.\n")
            continue

        break

    except ValueError:
        print("Please enter integer values only.\n")

# Calculate new dimensions
new_height = height + top + bottom
new_width = width + left + right

# Create a black image
padded = np.zeros((new_height, new_width, channels), dtype=np.uint8)

# Copy original image into padded image
for i in range(height):
    for j in range(width):
        padded[i + top, j + left] = image[i, j]


# Save
cv2.imwrite("padded_image.jpg", padded)
print("Padded image saved successfully.")

cv2.waitKey(0)
cv2.destroyAllWindows()