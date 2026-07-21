import cv2
import numpy as np
import matplotlib.pyplot as plt


# ================= MANUAL HISTOGRAM EQUALIZATION =================

def manual_histogram_equalization(grayscale_image):

    rows = grayscale_image.shape[0]
    cols = grayscale_image.shape[1]


    # Calculate histogram manually
    histogram = np.zeros(256, dtype=int)

    for i in range(rows):
        for j in range(cols):

            pixel = grayscale_image[i,j]

            histogram[pixel] += 1



    # Calculate CDF
    cdf = np.zeros(256, dtype=int)

    cdf[0] = histogram[0]

    for i in range(1,256):

        cdf[i] = cdf[i-1] + histogram[i]



    # Find minimum CDF value
    cdf_min = 0

    for i in range(256):

        if cdf[i] != 0:

            cdf_min = cdf[i]
            break



    total_pixels = rows * cols


    # Normalize CDF
    mapping = np.zeros(256, dtype=np.uint8)


    for i in range(256):

        mapping[i] = ((cdf[i]-cdf_min) /
                      (total_pixels-cdf_min)) * 255



    # Create equalized image

    equalized_image = np.zeros((rows,cols), dtype=np.uint8)


    for i in range(rows):
        for j in range(cols):

            old_pixel = grayscale_image[i,j]

            equalized_image[i,j] = mapping[old_pixel]


    return equalized_image



# ================= GRAYSCALE CONVERSION =================

def convert_to_grayscale(image):

    rows = image.shape[0]
    cols = image.shape[1]


    grayscale = np.zeros((rows,cols), dtype=np.uint8)


    for i in range(rows):
        for j in range(cols):

            b = image[i,j,0]
            g = image[i,j,1]
            r = image[i,j,2]


            grey = (0.299*r)+(0.587*g)+(0.114*b)


            grayscale[i,j] = grey


    return grayscale



# ================= PROCESS IMAGE =================

def process_image(image_path, name):


    image = cv2.imread(image_path)


    if image is None:
        print("Image not found:", image_path)
        return



    # Convert to grayscale
    grayscale = convert_to_grayscale(image)



    # Manual Histogram Equalization
    equalized = manual_histogram_equalization(grayscale)



    # CLAHE

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )


    clahe_image = clahe.apply(grayscale)



    # Save images

    cv2.imwrite(name+"_original.png", grayscale)

    cv2.imwrite(name+"_histogram_equalization.png",
                equalized)

    cv2.imwrite(name+"_clahe.png",
                clahe_image)



    # ================= COMPARISON PLOT =================

    plt.figure(figsize=(12,4))


    plt.subplot(1,3,1)

    plt.imshow(grayscale, cmap="gray")

    plt.title("Original")

    plt.axis("off")



    plt.subplot(1,3,2)

    plt.imshow(equalized, cmap="gray")

    plt.title("Histogram Equalization")

    plt.axis("off")



    plt.subplot(1,3,3)

    plt.imshow(clahe_image, cmap="gray")

    plt.title("CLAHE")

    plt.axis("off")



    plt.tight_layout()


    plt.savefig(name+"_comparison.png")

    plt.close()



# ================= RUN FOR THREE IMAGES =================


process_image(
    "dark_image_2.jpg",
    "dark"
)


process_image(
    "bright_image_2.jpg",
    "bright"
)


process_image(
    "low_contrast_image.jpg",
    "low_contrast"
)