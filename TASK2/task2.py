import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read the image using OpenCV
image = cv2.imread("../TASK1/image1.jpeg")

# Check if the image was loaded successfully
if image is None:
    print("Image not found!")
    exit()

#Getting the image dimensions
num_rows=image.shape[0]
num_col=image.shape[1]
num_channel=image.shape[2]

#Creating separate channel images
red_channel_image=np.zeros((num_rows,num_col),dtype=np.uint8)
green_channel_image=np.zeros((num_rows,num_col),dtype=np.uint8)
blue_channel_image=np.zeros((num_rows,num_col),dtype=np.uint8)

#Loop to separate all channels in image
for i in range(num_rows):
    for j in range(num_col):
        blue_channel_image[i][j]=image[i][j][0]
        green_channel_image[i][j]=image[i][j][1]
        red_channel_image[i][j]=image[i][j][2]

#Displaying blue channel image
plt.imshow(blue_channel_image,cmap="gray")
plt.savefig("blue_image.png")
print("Blue IMAGE CREATED SUCCESSFULLY")
plt.close()

#Displaying green channel image
plt.imshow(green_channel_image,cmap="gray")
plt.savefig("green_image.png")
print("Green IMAGE CREATED SUCCESSFULLY")
plt.close()

#Displaying red channel image
plt.imshow(red_channel_image,cmap="gray")
plt.savefig("red_image.png")
print("Red IMAGE CREATED SUCCESSFULLY")
plt.close()

#Creating merged image
merged_image=np.zeros((num_rows,num_col,num_channel),dtype=np.uint8)

#Loop to create merged image
for i in range(num_rows):
    for j in range(num_col):
        merged_image[i][j][0]=blue_channel_image[i][j]
        merged_image[i][j][1]=green_channel_image[i][j]
        merged_image[i][j][2]=red_channel_image[i][j]


#Before displaying the merged image we will convert it in rgb
for i in range(num_rows):
    for j in range(num_col):
        #logic to swap channels

        blue_value=merged_image[i][j][0]
        red_value=merged_image[i][j][2]
        merged_image[i][j][0]=red_value
        merged_image[i][j][2]=blue_value

#Displaying merged image
plt.imshow(merged_image)
plt.savefig("merged_image.png")
print("Merged IMAGE CREATED SUCCESSFULLY")
plt.close()