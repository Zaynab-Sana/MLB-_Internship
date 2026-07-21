import cv2
import numpy as np


image = cv2.imread("image.jpg")

if image is None:
    print("Image Not Found")
    exit()

#Get image dimensions
num_rows=image.shape[0]
num_col=image.shape[1]
channels=image.shape[2]

#CREATE EMPTY GREYSCALE IMAGE
greyscale_image=np.zeros((num_rows,num_col),dtype=np.uint8)

#LOOP TO CONVERT IN GREYSCALE IMAGE
for i in range(num_rows):
    for j in range(num_col):
        b=image[i,j,0]
        g=image[i,j,1]
        r=image[i,j,2]
        grey=(0.299*r)+(0.587*g)+(0.114*b)
        greyscale_image[i,j]=grey

#Select a threshold value
threshold=100

#===========================1-BINARY THRESHOLDING=====================================
#create an empty image
binary_threshold=np.zeros((num_rows,num_col),dtype=np.uint8)

#LOOP FOR THRESHOLDING
for i in range(num_rows):
    for j in range(num_col):
        if greyscale_image[i,j]>threshold:
            binary_threshold[i,j]=255
        else:
            binary_threshold[i,j]=0


#===========================2-BINARY INVERSE THRESHOLDING=====================================
#create an empty image
binary_inverse_threshold=np.zeros((num_rows,num_col),dtype=np.uint8)

#LOOP FOR THRESHOLDING
for i in range(num_rows):
    for j in range(num_col):
        if greyscale_image[i,j]>threshold:
            binary_inverse_threshold[i,j]=0
        else:
            binary_inverse_threshold[i,j]=255


#===========================3-TRUNCATE THRESHOLDING=====================================
#create an empty image
truncate_threshold=np.zeros((num_rows,num_col),dtype=np.uint8)

#LOOP FOR THRESHOLDING
for i in range(num_rows):
    for j in range(num_col):
        if greyscale_image[i,j]>threshold:
            truncate_threshold[i,j]=threshold
        else:
            truncate_threshold[i,j]=greyscale_image[i,j]

#===========================4-TO ZERO THRESHOLDING=====================================
#create an empty image
to_zero_threshold=np.zeros((num_rows,num_col),dtype=np.uint8)

#LOOP FOR THRESHOLDING
for i in range(num_rows):
    for j in range(num_col):
        if greyscale_image[i,j]>threshold:
            to_zero_threshold[i,j]=greyscale_image[i,j]
        else:
            to_zero_threshold[i,j]=0

#===========================5-INVERSE TO ZERO THRESHOLDING=====================================
#create an empty image
inverse_to_zero_threshold=np.zeros((num_rows,num_col),dtype=np.uint8)

#LOOP FOR THRESHOLDING
for i in range(num_rows):
    for j in range(num_col):
        if greyscale_image[i,j]>threshold:
            inverse_to_zero_threshold[i,j]=0
        else:
            inverse_to_zero_threshold[i,j]=greyscale_image[i,j]


#SAVING ALL IMAGES
cv2.imwrite("grayscale.png", greyscale_image)
cv2.imwrite("binary_threshold.png",binary_threshold)
cv2.imwrite("binary_inverse_threshold.png",binary_inverse_threshold)
cv2.imwrite("truncate_threshold.png",truncate_threshold)
cv2.imwrite("to_zero_threshold.png",to_zero_threshold)
cv2.imwrite("inverse_to_zero_threshold.png",inverse_to_zero_threshold)
