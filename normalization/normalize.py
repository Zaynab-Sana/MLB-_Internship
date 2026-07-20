import cv2
import numpy as np

image = cv2.imread("../reading_image/image.jpg")

if image is None:
    print("Image not found")
    exit()

height, width, channels = image.shape

# Create an empty floating-point image
normalized = np.zeros((height, width, channels), dtype=np.float32)

# Manual normalization
for i in range(height):
    for j in range(width):
        for k in range(channels):
            normalized[i, j, k] = image[i, j, k] / 255.0

print("Original Pixel:", image[0, 0])
print("Normalized Pixel:", normalized[0, 0])


# Convert normalized image back to 0–255 for display
display_image = (normalized * 255).astype(np.uint8)



# Save the  image
cv2.imwrite("normalized_image.jpg", display_image)

cv2.waitKey(0)
cv2.destroyAllWindows()