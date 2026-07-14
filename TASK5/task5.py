#===============TASK 5 =======================
import cv2
import numpy as np

#load image
image=cv2.imread("../TASK2/modifiedimage2.png")

#Validation
if image is None:
    print("Image Not found")
    exit()

# Get image dimensions
num_rows = image.shape[0]
num_col = image.shape[1]

# Create empty images for each channel
blue_image = np.zeros((num_rows, num_col), dtype=np.uint8)

green_image = np.zeros((num_rows, num_col), dtype=np.uint8)

red_image = np.zeros((num_rows, num_col), dtype=np.uint8)

#USE LOOP TO CREATE NEW IMAGES
for i in range(num_rows):
    for j in range(num_col):
                blue_image[i][j]=image[i][j][0]
                green_image[i][j]=image[i][j][1]
                red_image[i][j]=image[i][j][2]


cv2.imwrite("blueimage.png",blue_image)
cv2.imwrite("greenimage.png",green_image)
cv2.imwrite("redimage.png",red_image)




