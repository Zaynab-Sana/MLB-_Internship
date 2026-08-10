import cv2
import numpy as np
import matplotlib.pyplot as plt




# =========================================================
# FUNCTION: CALCULATE HISTOGRAM MANUALLY
# =========================================================

def calculate_histogram(image):

    histogram = np.zeros(256, dtype=int)

    for pixel in image.flatten():
        histogram[pixel] += 1

    return histogram


# =========================================================
# FUNCTION: MANUAL HISTOGRAM EQUALIZATION USING CDF
# =========================================================

def histogram_equalization(image):

    # Calculate histogram
    histogram = calculate_histogram(image)

    # Calculate CDF
    cdf = np.cumsum(histogram)

    # First non-zero CDF value
    cdf_min = cdf[cdf > 0][0]

    # Total number of pixels
    total_pixels = image.size

    # CDF normalization
    mapping = (
        (cdf - cdf_min)
        / (total_pixels - cdf_min)
        * 255
    )

    # Convert mapping to uint8
    mapping = np.round(mapping).astype(np.uint8)

    # Apply mapping
    equalized_image = mapping[image]

    return equalized_image


# =========================================================
# THREE LOW-CONTRAST IMAGES
# =========================================================

image_paths = [
    "lowcontrast1.jpg",
    "lowcontrast2.jpg",
    "lowcontrast3.png"
]

image_names = [
    "Low Contrast Image 1",
    "Low Contrast Image 2",
    "Low Contrast Image 3"
]


# =========================================================
# CREATE CLAHE
# =========================================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)


# =========================================================
# PROCESS ALL THREE IMAGES
# =========================================================

for path, name in zip(image_paths, image_names):

    # -----------------------------------------------------
    # Read image as grayscale
    # -----------------------------------------------------

    image = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        print("Error: Could not read:", path)

        continue


    # -----------------------------------------------------
    # Normal Histogram Equalization
    # -----------------------------------------------------

    equalized_image = histogram_equalization(image)


    # -----------------------------------------------------
    # CLAHE
    # -----------------------------------------------------

    clahe_image = clahe.apply(image)


    # -----------------------------------------------------
    # Calculate Histograms
    # -----------------------------------------------------

    original_histogram = calculate_histogram(
        image
    )

    equalized_histogram = calculate_histogram(
        equalized_image
    )

    clahe_histogram = calculate_histogram(
        clahe_image
    )


    # =====================================================
    # CREATE A SEPARATE FIGURE
    # =====================================================

    fig = plt.figure(
        figsize=(16, 9),
        num=name
    )


    # =====================================================
    # TOP ROW - IMAGES
    # =====================================================

    # Original
    ax1 = fig.add_subplot(2, 3, 1)

    ax1.imshow(
        image,
        cmap="gray"
    )

    ax1.set_title(
        "Original",
        fontsize=14,
        fontweight="bold"
    )

    ax1.axis("off")


    # Histogram Equalization
    ax2 = fig.add_subplot(2, 3, 2)

    ax2.imshow(
        equalized_image,
        cmap="gray"
    )

    ax2.set_title(
        "Histogram Equalization",
        fontsize=14,
        fontweight="bold"
    )

    ax2.axis("off")


    # CLAHE
    ax3 = fig.add_subplot(2, 3, 3)

    ax3.imshow(
        clahe_image,
        cmap="gray"
    )

    ax3.set_title(
        "CLAHE",
        fontsize=14,
        fontweight="bold"
    )

    ax3.axis("off")


    # =====================================================
    # BOTTOM ROW - HISTOGRAMS
    # =====================================================

    # Original Histogram
    ax4 = fig.add_subplot(2, 3, 4)

    ax4.bar(
        np.arange(256),
        original_histogram,
        width=1
    )

    ax4.set_title(
        "Original Histogram",
        fontsize=13,
        fontweight="bold"
    )

    ax4.set_xlabel(
        "Pixel Intensity (0-255)"
    )

    ax4.set_ylabel(
        "Frequency"
    )

    ax4.set_xlim(0, 255)


    # Equalized Histogram
    ax5 = fig.add_subplot(2, 3, 5)

    ax5.bar(
        np.arange(256),
        equalized_histogram,
        width=1
    )

    ax5.set_title(
        "Equalized Histogram",
        fontsize=13,
        fontweight="bold"
    )

    ax5.set_xlabel(
        "Pixel Intensity (0-255)"
    )

    ax5.set_ylabel(
        "Frequency"
    )

    ax5.set_xlim(0, 255)


    # CLAHE Histogram
    ax6 = fig.add_subplot(2, 3, 6)

    ax6.bar(
        np.arange(256),
        clahe_histogram,
        width=1
    )

    ax6.set_title(
        "CLAHE Histogram",
        fontsize=13,
        fontweight="bold"
    )

    ax6.set_xlabel(
        "Pixel Intensity (0-255)"
    )

    ax6.set_ylabel(
        "Frequency"
    )

    ax6.set_xlim(0, 255)


    # =====================================================
    # FIGURE TITLE
    # =====================================================

    fig.suptitle(
        f"Task 5: {name}",
        fontsize=18,
        fontweight="bold"
    )


    # Adjust layout
    fig.tight_layout(
        rect=[0, 0, 1, 0.94]
    )


# =========================================================
# SHOW ALL THREE FIGURES AT ONCE
# =========================================================

plt.show()