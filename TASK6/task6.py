#===============TASK 6 =======================
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
num_channel=image.shape[2]

# Create empty images for each channel
grey_scale_image = np.zeros((num_rows, num_col), dtype=np.uint8)


#USE LOOP TO CREATE NEW IMAGE
for i in range(num_rows):
    for j in range(num_col):
        channel_sum = 0
        for k in range(num_channel):
            value = int(image[i][j][k])
            channel_sum += value
        value = channel_sum / num_channel
        grey_scale_image[i][j]=value



cv2.imwrite("greyscaleimage.png",grey_scale_image)





