import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read the image using OpenCV
image = cv2.imread("../TASK1/image1.jpeg")

# Check if the image was loaded successfully
if image is None:
    print("Image not found!")
    exit()

#===================================1-CONVERSION TO RGB===========================================
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



#=======================================2-CONVERSION TO HSL AND HSV================================

# Create empty hsv and hsl images
hsv_image = np.zeros((num_rows,num_col, 3), dtype=np.uint8)
hsl_image = np.zeros((num_rows,num_col, 3), dtype=np.uint8)

#Using loop to get channel values
for i in range(num_rows):
    for j in range(num_col):

        # Get BGR values and normalize to 0-1
        b = image[i][j][0] / 255.0
        g = image[i][j][1] / 255.0
        r = image[i][j][2] / 255.0

        cmax = max(r, g, b)
        cmin = min(r, g, b)
        delta = cmax - cmin

        # 1-Formula to calculate hue in hsv,hsl
        if delta == 0:
            h = 0
        elif cmax == r:
            h = 60 * (((g - b) / delta) % 6)
        elif cmax == g:
            h = 60 * (((b - r) / delta) + 2)
        else:
            h = 60 * (((r - g) / delta) + 4)

        # 2-Formula to calculate saturation in hsv
        if cmax == 0:
            s_hsv = 0
        else:
            s_hsv = delta / cmax

         #3- Formula to calculate value in hsv
        v = cmax

        # 4-Formula to calculate lightness in hsl
        l = (cmax + cmin) / 2

        #Formula to calculate saturation in hsl
        if delta == 0:
            s_hsl = 0
        else:
            s_hsl = delta / (1 - abs(2 * l - 1))

        # Storing channel values in hsv image
        hsv_image[i][j][0] = h / 2
        hsv_image[i][j][1] = s_hsv * 255
        hsv_image[i][j][2] = v * 255

        # Storing channel values in hsl image
        hsl_image[i][j][0] = h / 2
        hsl_image[i][j][1] = l * 255
        hsl_image[i][j][2] = s_hsl * 255

#===============================3-CONVERSION TO LAB IMAGE==================================
# Create empty LAB image
lab_image = np.zeros((num_rows, num_col, 3), dtype=np.uint8)

# Reference white
Xn = 0.95047
Yn = 1.00000
Zn = 1.08883

for i in range(num_rows):
    for j in range(num_col):

       #Normalize values
        b = image[i][j][0] / 255.0
        g = image[i][j][1]/ 255.0
        r = image[i][j][2] / 255.0

        # Gamma correction
        if r > 0.04045:
            r = ((r + 0.055) / 1.055) ** 2.4
        else:
            r = r / 12.92

        if g > 0.04045:
            g = ((g + 0.055) / 1.055) ** 2.4
        else:
            g = g / 12.92

        if b > 0.04045:
            b = ((b + 0.055) / 1.055) ** 2.4
        else:
            b = b / 12.92


        # Convert RGB to XYZ
        X = 0.4124 * r + 0.3576 * g + 0.1805 * b
        Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        Z = 0.0193 * r + 0.1192 * g + 0.9505 * b

        # Normalize
        x = X / Xn
        y = Y / Yn
        z = Z / Zn

        # f(t)
        if x > 0.008856:
            fx = x ** (1/3)
        else:
            fx = (7.787 * x) + (16 / 116)

        if y > 0.008856:
            fy = y ** (1/3)
        else:
            fy = (7.787 * y) + (16 / 116)

        if z > 0.008856:
            fz = z ** (1/3)
        else:
            fz = (7.787 * z) + (16 / 116)

        # LAB
        L = (116 * fy) - 16
        A = 500 * (fx - fy)
        B = 200 * (fy - fz)

        # Scale for display
        L = int((L / 100) * 255)
        A = int(A + 128)
        B = int(B + 128)

        # Clip values to 0-255
        L = max(0, min(255, L))
        A = max(0, min(255, A))
        B = max(0, min(255, B))

        lab_image[i, j] = [L, A, B]

#============================4-CONVERSION TO GREYSCALE IMAGE======================================
#Creating empty grey image
gray_image = np.zeros((num_rows, num_col), dtype=np.uint8)


#Loop to find blue, green , red values
for i in range(num_rows):
    for j in range(num_col):

        b = image[i][j][0]
        g = image[i][j][1]
        r = image[i][j][2]

        #Applying standard formula
        gray = int(0.299 * r + 0.587 * g + 0.114 * b)

        gray_image[i][j] = gray

#===========================DISPLAYING ALL IMAGES========================================
plt.figure(figsize=(15,10))

plt.subplot(2,3,1)
plt.imshow(rgb_image)
plt.title("RGB")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(hsv_image)
plt.title("HSV")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(hsl_image)
plt.title("HSL")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(lab_image)
plt.title("LAB")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(gray_image, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(rgb_image)
plt.title("Original RGB")
plt.axis("off")

plt.tight_layout()

plt.savefig("task3_output.png", dpi=300, bbox_inches="tight")
plt.close()

print("Figure saved successfully as task3_output.png")

'''
Here are the uses of each color space in simple paragraph form:

1-RGB (Red, Green, Blue)

RGB is the most common color space used to display images on electronic devices such as computer monitors, mobile phones, televisions, and digital cameras. It represents colors by combining different amounts of red, green, and blue light.

2-HSV (Hue, Saturation, Value)

HSV is mainly used in computer vision because it separates color information from brightness. This makes it easier to detect and segment objects based on their color, even when lighting conditions change. It is commonly used for color detection, object tracking, and image segmentation.

3-HSL (Hue, Saturation, Lightness)

HSL is commonly used in graphic design and image editing applications. It separates the color from its lightness, making it easier to adjust brightness and create visually pleasing color variations without changing the actual color.

4-LAB (Lightness, A, B)

LAB is designed to represent colors in a way that closely matches human vision. It separates brightness (Lightness) from color information (A and B channels). It is widely used for image enhancement, color correction, color comparison, medical imaging, and printing because it provides more accurate color representation.

5-Grayscale

Grayscale images contain only brightness information and no color. They are widely used in image processing tasks because they reduce the amount of data while preserving important details. Grayscale images are commonly used for edge detection, face recognition, OCR (Optical Character Recognition), and other computer vision applications.

6-Binary Image

A binary image contains only two pixel values: black and white. It is mainly used to separate objects from the background after thresholding. Binary images are commonly used in document scanning, object detection, shape analysis, character recognition, and image segmentation.
'''