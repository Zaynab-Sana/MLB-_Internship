import cv2
import matplotlib.pyplot as plt

# Read image
image = cv2.imread("image2.jpg")

if image is None:
    print("Image not found!")
    exit()

# Get image dimensions
height, width, channels = image.shape

# creating a copy of image
marked_image = image.copy()

# asking user for marking coordinates
num_points = int(input("Enter the number of coordinates to mark: "))

for i in range(num_points):

    print(f"\nCoordinate {i + 1}")

    x = int(input(f"Enter X coordinate (0 - {width - 1}): "))
    y = int(input(f"Enter Y coordinate (0 - {height - 1}): "))

    # Coordinate validation
    if x < 0 or x >= width or y < 0 or y >= height:
        print("Invalid coordinates! Please try again.")
        continue

    # Print pixel value
    print(f"Pixel Value at ({x}, {y}) = {image[y][x]}")

    # Draw a red circle
    cv2.circle(marked_image, (x, y), 5, (0, 0, 255), -1)

    # Write coordinates beside the point
    cv2.putText(
        marked_image,
        f"({x},{y})",
        (x + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1
    )

# Create a copy of the image
rgb_image = marked_image.copy()

# Manual BGR to RGB conversion
for row in range(height):
    for col in range(width):

        blue = marked_image[row][col][0]
        green = marked_image[row][col][1]
        red = marked_image[row][col][2]

        rgb_image[row][col][0] = red
        rgb_image[row][col][1] = green
        rgb_image[row][col][2] = blue

# Display image
plt.figure(figsize=(8, 8))
plt.imshow(rgb_image)
plt.title("Image with Marked Coordinates")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)

# Save the figure
plt.savefig("marked_coordinates.png", dpi=300, bbox_inches="tight")

# Close the figure
plt.close()


#explanation
print("\nDifference Between Coordinate System and Matrix Indexing")
print("--------------------------------------------------------")
print("Image Coordinate System : (x, y)")
print("Matrix Indexing         : image[row][column]")
print("x represents the column.")
print("y represents the row.")
print("To access coordinate (x, y), use image[y][x].")