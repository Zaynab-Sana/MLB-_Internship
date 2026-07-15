import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read Image using open cv
image = cv2.imread("../TASK1/image1.jpeg")

#Validation
if image is None:
    print("Image not found!")
    exit()

rows=image.shape[0]
cols=image.shape[1]
channels=image.shape[2]

# First create an empty image
lab_image = np.zeros((rows, cols, 3), dtype=np.float32)

# Reference white (D65)
Xn = 95.047
Yn = 100.000
Zn = 108.883

for i in range(rows):
    for j in range(cols):

        # normalize values
        B = image[i][j][0] / 255.0
        G = image[i][j][1] / 255.0
        R = image[i][j][2] / 255.0

        # Gamma Correction
        if R > 0.04045:
            R = ((R + 0.055) / 1.055) ** 2.4
        else:
            R = R / 12.92

        if G > 0.04045:
            G = ((G + 0.055) / 1.055) ** 2.4
        else:
            G = G / 12.92

        if B > 0.04045:
            B = ((B + 0.055) / 1.055) ** 2.4
        else:
            B = B / 12.92

        # RGB → XYZ
        X = (0.4124564 * R +
             0.3575761 * G +
             0.1804375 * B) * 100

        Y = (0.2126729 * R +
             0.7151522 * G +
             0.0721750 * B) * 100

        Z = (0.0193339 * R +
             0.1191920 * G +
             0.9503041 * B) * 100


        # Normalize XYZ
        X = X / Xn
        Y = Y / Yn
        Z = Z / Zn

        # XYZ → f(t)
        if X > 0.008856:
            fx = X ** (1/3)
        else:
            fx = (7.787 * X) + (16/116)

        if Y > 0.008856:
            fy = Y ** (1/3)
        else:
            fy = (7.787 * Y) + (16/116)

        if Z > 0.008856:
            fz = Z ** (1/3)
        else:
            fz = (7.787 * Z) + (16/116)

        # LAB Calculation
        L = (116 * fy) - 16
        A = 500 * (fx - fy)
        B_lab = 200 * (fy - fz)

        lab_image[i][j][0] = L
        lab_image[i][j][1] = A
        lab_image[i][j][2] = B_lab

#separate lab channels
L_channel = lab_image[:, :, 0]
A_channel = lab_image[:, :, 1]
B_channel = lab_image[:, :, 2]

# Normalize channels for display
L_display = (L_channel / 100) * 255

A_display = ((A_channel + 128) / 255) * 255
B_display = ((B_channel + 128) / 255) * 255

L_display = np.clip(L_display, 0, 255).astype(np.uint8)
A_display = np.clip(A_display, 0, 255).astype(np.uint8)
B_display = np.clip(B_display, 0, 255).astype(np.uint8)

# Manual BGR → RGB (for display)
rgb_image = image.copy()

for i in range(rows):
    for j in range(cols):

        temp = rgb_image[i][j][0]
        rgb_image[i][j][0] = rgb_image[i][j][2]
        rgb_image[i][j][2] = temp

# Display
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(rgb_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(L_display, cmap="gray")
plt.title("L Channel")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(A_display, cmap="gray")
plt.title("A Channel")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(B_display, cmap="gray")
plt.title("B Channel")
plt.axis("off")

plt.tight_layout()
plt.savefig("task5_lab_channels.png", bbox_inches="tight")

plt.close()