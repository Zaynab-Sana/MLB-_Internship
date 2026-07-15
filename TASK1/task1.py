import cv2
import matplotlib.pyplot as plt

# Read the image using OpenCV (OpenCV reads images in BGR format)
image = cv2.imread("image1.jpeg")

# Check if the image was loaded successfully
if image is None:
    print("Image not found!")
    exit()

#Using matplotlib to display image, here it stores incorrect colors as it expect rbg instead of bgr color space
plt.imshow(image)
plt.savefig("bgr_image.png")
print("BGR INCORRECT IMAGE CREATED SUCCESSFULLY")
plt.close()

#Get image dimensions
num_rows=image.shape[0]
num_col=image.shape[1]
num_channel =image.shape[2]

#Creating a copy of original image
rgb_image=image.copy()

#USING LOOP TO CONVERT THE IMAGE FROM BGR TO RGB
for i in range(num_rows):
    for j in range(num_col):
        #logic to swap channels

        blue_value=image[i][j][0]
        red_value=image[i][j][2]
        rgb_image[i][j][0]=red_value
        rgb_image[i][j][2]=blue_value


#Showing rgb image
plt.imshow(rgb_image)
plt.savefig("rgb_image.png")
print("BGR IMAGE Converted to RGB SUCCESSFULLY")

#EXplanation
'''
OpenCV reads images in BGR (Blue, Green, Red) format because of historical reasons and compatibility with older computer vision systems.

Most libraries like Matplotlib use RGB (Red, Green, Blue) format. Since the order of the red and blue channels is reversed, directly displaying an OpenCV image may show incorrect colors.

To display the image correctly, the Blue and Red channels are swapped:

BGR → RGB

The Green channel remains the same in both formats.
'''








