import cv2


image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found!")
    exit()


print("Image Matrix:\n")
print(image)

#print image dimensions
height, width = image.shape

print("\nImage Dimensions:")
print(f"Height = {height}")
print(f"Width  = {width}")

#Printing pixel values
print("\nPixel Values:")

#loop to print only first 5 values
for row in range(5):
    for col in range(5):
        print(f"Pixel at ({col}, {row}) = {image[row][col]}")

#modifying pixel values

while True:

    print("\nEnter the coordinate of the pixel you want to modify.")

    x = int(input(f"Enter X (0 - {width-1}): "))
    y = int(input(f"Enter Y (0 - {height-1}): "))

    # Coordinate validation
    if x < 0 or x >= width or y < 0 or y >= height:
        print("Invalid coordinates! Please enter valid values.\n")
        continue

    new_value = int(input("Enter new pixel intensity (0 - 255): "))

    # Intensity validation
    if new_value < 0 or new_value > 255:
        print("Invalid intensity! Enter a value between 0 and 255.\n")
        continue

    # Update pixel
    image[y][x] = new_value

    print(f"\nPixel at ({x}, {y}) updated successfully!")


    choice = input("\nDo you want to modify another pixel? (y/n): ").lower()

    if choice != 'y':
        break

#saving modified image

cv2.imwrite("modified_image.jpg", image)

cv2.waitKey(0)
cv2.destroyAllWindows()