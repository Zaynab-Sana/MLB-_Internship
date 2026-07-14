#===============TASK 4 =======================
import cv2

#load image
image=cv2.imread("../TASK2/modifiedimage2.png")

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
blackpiexel_count=0
whitepiexel_count=0
bright_pixel_count=0


#Loop to calculate piexel intensities
for i in range(num_rows):
    for j in range(num_col):
        channel_sum=0
        for k in range(num_channel):
            value=int(image[i][j][k])
            channel_sum+=value

            #Logic to count black and white piexels
        if channel_sum==0:
            blackpiexel_count+=1
        if channel_sum==255*num_channel:
            whitepiexel_count+=1

        value=channel_sum/num_channel

        #LOgic to count piexels which have intensity greater than 200
        if value > 200:
            bright_pixel_count+=1

print("Total pixel count:",total_pixel)
print("Black pixel count:",blackpiexel_count)
print("White pixel count:",whitepiexel_count)
print("Pixels having intensity greater than 200:" ,bright_pixel_count)





