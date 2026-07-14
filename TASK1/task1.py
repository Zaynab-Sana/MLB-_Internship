#==============TASK 1==============

import cv2

#load image
image=cv2.imread("./image1.jpg")
#validation
if image is None:
    print("Image Not Found")
    exit()

#Storing values in variables
height=image.shape[0]
width=image.shape[1]
channels=image.shape[2]

#Printing values
print("Height:",height)
print("Width:",width)
print("Channels:",channels)

#Printing datatype
datatype=image.dtype
print("\nData Type of Image:",datatype)

#Calculating Total Piexels
totalPixels=height*width
print("\nTotal Pixels:",totalPixels)