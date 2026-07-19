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

#loop to enter and validate the cropping information
while True:
    print("Enter the information for cropping image:")

    print(f"Enter the start row (0 - {num_rows - 1}):")
    start_row = int(input())

    print(f"Enter the end row ({start_row + 1} - {num_rows-1}):")
    end_row = int(input())

    print(f"Enter the start column (0 - {num_col - 1}):")
    start_col = int(input())

    print(f"Enter the end column ({start_col + 1} - {num_col-1}):")
    end_col = int(input())

    #Validating the input
    if (start_row < 0 or start_row >= num_rows or
        end_row <= start_row or end_row >= num_rows or
        start_col < 0 or start_col >= num_col or
        end_col <= start_col or end_col >= num_col):
        print("Invalid values entered. Please try again.\n")
    else:
        break

#new dimensions for cropped image
crop_image_rows=end_row-start_row + 1
crop_image_cols=end_col-start_col + 1

#create an empty image
cropped_image=np.zeros((crop_image_rows,crop_image_cols,channels),dtype=np.uint8)

#loop to copy pixels of original image in cropped image
for i in range(crop_image_rows):
    for j in range(crop_image_cols):
        cropped_image[i,j,0]=image[start_row+i,start_col+j,0]
        cropped_image[i,j,1]=image[start_row+i,start_col+j,1]
        cropped_image[i,j,2]=image[start_row+i,start_col+j,2]


saved=cv2.imwrite("cropped_image.png",cropped_image)
if saved:
    print ("Image saved successfully")
else:
    print ("Image not saved")

