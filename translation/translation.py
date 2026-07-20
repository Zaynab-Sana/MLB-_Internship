import cv2
import numpy as np

image = cv2.imread("../reading_image/image.jpg")

if image is None:
    print("Image not found")
    exit()

height, width, channels = image.shape

# Printing translation instructions
print("Translation Instructions:")
print("+X : Move Right")
print("-X : Move Left")
print("+Y : Move Down")
print("-Y : Move Up")
print()

# User input validation
while True:
    try:
        tx = int(input("Enter translation in X direction: "))
        ty = int(input("Enter translation in Y direction: "))

        if tx < -width + 1 or tx > width - 1:
            print(f"X translation should be between {-width + 1} and {width - 1}\n")
            continue

        if ty < -height + 1 or ty > height - 1:
            print(f"Y translation should be between {-height + 1} and {height - 1}\n")
            continue

        break

    except ValueError:
        print("Please enter integer values only.\n")

# Create an empty image
translated = np.zeros((height, width, channels), dtype=np.uint8)

# Manual translation
for y in range(height):
    for x in range(width):

        new_x = x + tx
        new_y = y + ty

        if 0 <= new_x < width and 0 <= new_y < height:
            translated[new_y, new_x] = image[y, x]


# Save translated image
cv2.imwrite("translated_image.jpg", translated)
print("Translated image saved successfully.")

cv2.waitKey(0)
cv2.destroyAllWindows()