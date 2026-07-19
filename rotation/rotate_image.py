import cv2
import numpy as np

image = cv2.imread("../reading_image/image.jpg")

if image is None:
    print("Image Not Found")
    exit()

#getting dimensions
num_rows = image.shape[0]
num_cols = image.shape[1]
channels = image.shape[2]

#loop to select rotation type
while True:
    print("Select Rotation:")
    print("1. Rotate 90° Clockwise")
    print("2. Rotate 180°")
    print("3. Rotate 270° Clockwise")
    choice = int(input("Enter your choice: "))

    #validation
    if choice in [1, 2, 3]:
        break
    else:
        print("Invalid Choice! Try Again.\n")


#90 degree
if choice == 1:

    #creating an empty image
    rotated_image = np.zeros((num_cols, num_rows, channels), dtype=np.uint8)

    #changing pixel positions
    for i in range(num_rows):
        for j in range(num_cols):
            rotated_image[j, num_rows - 1 - i] = image[i, j]


# 180 Degree
elif choice == 2:

    # creating an empty image
    rotated_image = np.zeros((num_rows, num_cols, channels), dtype=np.uint8)

    for i in range(num_rows):
        for j in range(num_cols):
            rotated_image[num_rows - 1 - i, num_cols - 1 - j] = image[i, j]


# 270 Degree Clockwise
elif choice == 3:

    # creating an empty image
    rotated_image = np.zeros((num_cols, num_rows, channels), dtype=np.uint8)

    for i in range(num_rows):
        for j in range(num_cols):
            rotated_image[num_cols - 1 - j, i] = image[i, j]


saved = cv2.imwrite("rotated_image.png", rotated_image)

if saved:
    print("Image Saved Successfully")
else:
    print("Image Not Saved")




