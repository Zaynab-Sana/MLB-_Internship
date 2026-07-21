import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread("../TASK 1/image.jpg")

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


#create an empty array for histogram values
histogram=np.zeros(256,dtype=int)

for i in range(num_rows):
    for j in range(num_col):
        value=greyscale_image[i,j]
        histogram[value]=histogram[value]+1

# Create intensity values (0 to 255)
intensity = np.arange(256)

# Plot histogram
plt.plot(intensity, histogram)

# Add title and labels
plt.title("Histogram of Grayscale Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

# Set x-axis range
plt.xlim([0, 255])

# Save the histogram
plt.savefig("histogram.png")

# Close the figure
plt.close()
