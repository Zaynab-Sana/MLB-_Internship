import cv2

image=cv2.imread('image.jpg')
if image is None:
    print ("Image Not Found")
    exit()
else:
    print ("Image read successfully")
    print("Image information:")
    print("Height:",image.shape[0])
    print("Width:",image.shape[1])
    print("Channels:",image.shape[2])