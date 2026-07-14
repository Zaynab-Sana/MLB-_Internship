#=============TASK 2=================

import cv2

#load image
image=cv2.imread("../TASK1/image1.jpg")

if image is None:
    print("Image Not found")
    exit()

height=image.shape[0]
width=image.shape[1]

for i in range(5):

    print(f"\nPixel {i+1}")

    while True:
        row = int(input("Enter row: "))
        col = int(input("Enter column: "))

        # Validate coordinates
        if 0 <= row < height and 0 <= col < width:
            break
        else:
            print("Invalid coordinates! Please try again.")

    # Enter new pixel values
    blue = int(input("Enter Blue value (0-255): "))
    green = int(input("Enter Green value (0-255): "))
    red = int(input("Enter Red value (0-255): "))

    # Validate pixel values
    if not (0 <= blue <= 255 and 0 <= green <= 255 and 0 <= red <= 255):
        print("Invalid color values! Pixel not changed.")
        continue

    # Change pixel
    image[row][col] = [blue, green, red]

    print(f"Pixel ({row}, {col}) changed successfully.")

#Saving the image
cv2.imwrite("modifiedimage2.png",image)

print("\nModified image saved successfully as 'modified_image.jpg'")
