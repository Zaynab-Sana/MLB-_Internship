import cv2
import numpy as np

image = cv2.imread("../reading_image/image.jpg")

if image is None:
    print("Image not found")
    exit()

height, width, channels = image.shape

# User input with validation
while True:
    try:
        sx = float(input("Enter scale factor for X (e.g., 2 or 0.5): "))
        sy = float(input("Enter scale factor for Y (e.g., 2 or 0.5): "))

        if sx <= 0 or sy <= 0:
            print("Scale factors must be greater than 0.\n")
            continue

        break

    except ValueError:
        print("Please enter valid numeric values.\n")

# Calculate new dimensions
new_width = int(width * sx)
new_height = int(height * sy)

# Create an empty image
scaled = np.zeros((new_height, new_width, channels), dtype=np.uint8)

# loop for scaling
for i in range(new_height):
    for j in range(new_width):

        old_x = int(j / sx)
        old_y = int(i / sy)

        # Boundary check
        if old_x >= width:
            old_x = width - 1

        if old_y >= height:
            old_y = height - 1

        scaled[i, j] = image[old_y, old_x]


# Save
cv2.imwrite("scaled_image.jpg", scaled)
print("Scaled image saved successfully.")

cv2.waitKey(0)
cv2.destroyAllWindows()