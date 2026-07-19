import cv2
import numpy as np


image=cv2.imread("../reading_image/image.jpg")
if image is None:
    print ("Image Not Found")
    exit()

#getting the image dimension
num_rows=image.shape[0]
num_col=image.shape[1]
channels=image.shape[2]

#Printing previous height and width
print(f"Previous height: {num_rows}")
print(f"Previous width: {num_col}")


#ASKING USER TO ENTER INFORMATION FOR RESIZING
while True:
    print("Enter the new height (0-any):")
    height=int(input())
    print("Enter the new width (0-any):")
    width=int(input())
    if (height<=0 or width<=0):
        print("Invalid information enter again!")
    else:
        break

#creating an empty image
resized_image=np.zeros((height,width,channels),dtype=np.uint8)

#calculating ratios
row_ratio=num_rows/height
column_ratio=num_col/width

#loop to copy pixels in resized image
for i in range(height):
    for j in range(width):
        original_row = int(i * row_ratio)
        original_col = int(j * column_ratio)

        resized_image[i, j, 0] = image[original_row, original_col, 0]
        resized_image[i, j, 1] = image[original_row, original_col, 1]
        resized_image[i, j, 2] = image[original_row, original_col, 2]

saved=cv2.imwrite("resized_image.png",resized_image)
if saved:
    print ("Image saved")
else:
    print("Image Not Saved")