import cv2
import numpy as np


image=cv2.imread("../reading_image/image.jpg")
if image is None:
    print ("Image Not Found")
    exit()

num_rows=image.shape[0]
num_col=image.shape[1]
channels=image.shape[2]

copied_image=np.zeros((num_rows,num_col,channels),dtype=np.uint8)

for i in range(num_rows):
    for j in range(num_col):
        copied_image[i][j][0]=image[i][j][0]
        copied_image[i][j][1]=image[i][j][1]
        copied_image[i][j][2]=image[i][j][2]

saved=cv2.imwrite("copied_image.jpg",copied_image)
if saved:
    print ("Image Copied successfully")
else:
    print ("Image Not Copied successfully")