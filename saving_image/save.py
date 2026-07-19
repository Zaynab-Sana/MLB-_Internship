import cv2

image=cv2.imread("../reading_image/image.jpg")
if image is None:
    print ("Image Not Found")
    exit()

saved=cv2.imwrite("saved_image.jpg",image)
if saved:
    print ("Image Saved successfully")
else:
    print ("Image Not Saved")