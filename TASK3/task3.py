#==============TASK 3==============

import cv2

#load image
image=cv2.imread("../TASK1/image1.jpg")

#Validation
if image is None:
    print("Image Not found")
    exit()

#Finding number of rows, columns, and channel
num_rows=image.shape[0]
num_col=image.shape[1]
num_channel=image.shape[2]

#Formula to calculate total piexels
total_pixel=num_rows*num_col

total_intensity=0
min_intensity=255
max_intensity=0

#Loop to calculate sum of piexel intensities
for i in range(num_rows):
    for j in range(num_col):
        channel_sum=0
        for k in range(num_channel):
            value=int(image[i][j][k])
            channel_sum+=value
        value=channel_sum/num_channel

        #LOGIC TO FIND MINIMUM AND MAXIMUM VALUE
        if value<min_intensity:
            min_intensity=value
        if value>max_intensity:
            max_intensity=value
        total_intensity+=value
        #LOOP END

mean_intensity=total_intensity/total_pixel
print (f"Mean Intensity :{mean_intensity:.2f} ")
print (f"Max Intensity :{max_intensity:.2f} ")
print (f"Min Intensity :{min_intensity:.2f} ")

