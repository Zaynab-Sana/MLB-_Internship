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
# FUNCTION: HISTOGRAM EQUALIZATION USING CDF
# =========================================================

def histogram_equalization(image):

    histogram = calculate_histogram(image)

    cdf = np.cumsum(histogram)

    cdf_min = cdf[cdf > 0][0]

    total_pixels = image.size

    mapping = (
        (cdf - cdf_min)
        / (total_pixels - cdf_min)
        * 255
    )

    mapping = np.round(mapping).astype(np.uint8)

    equalized_image = mapping[image]

    return equalized_image


# =========================================================
# FUNCTION: CALCULATE NORMALIZED CDF
# =========================================================

def calculate_cdf(histogram):

    cdf = np.cumsum(histogram)

    cdf = cdf / cdf[-1]

    return cdf


# =========================================================
# FUNCTION: HISTOGRAM MATCHING USING CDF
# =========================================================

def histogram_matching(source, reference):

    # Source histogram
    source_histogram = calculate_histogram(
        source
    )

    # Reference histogram
    reference_histogram = calculate_histogram(
        reference
    )

    # Source CDF
    source_cdf = calculate_cdf(
        source_histogram
    )

    # Reference CDF
    reference_cdf = calculate_cdf(
        reference_histogram
    )

    # Mapping table
    mapping = np.zeros(
        256,
        dtype=np.uint8
    )

    # Find closest reference CDF value
    for source_intensity in range(256):

        difference = np.abs(
            reference_cdf
            - source_cdf[source_intensity]
        )

        reference_intensity = np.argmin(
            difference
        )

        mapping[source_intensity] = (
            reference_intensity
        )

    # Apply mapping
    matched_image = mapping[source]

    return matched_image


# =========================================================
# FOUR INPUT IMAGES
# =========================================================

image_paths = [
    "bright.jpg",
    "dark.jpg",
    "highcontrast.jpg",
    "lowcontrast.jpg"
]

image_names = [
    "Image 1",
    "Image 2",
    "Image 3",
    "Image 4"
]


# =========================================================
# REFERENCE IMAGE
# =========================================================

reference = cv2.imread(
    "reference.jpg",
    cv2.IMREAD_GRAYSCALE
)


# =========================================================
# CHECK REFERENCE IMAGE
# =========================================================

if reference is None:

    print("Error: Could not read reference.jpg")

    exit()


# =========================================================
# DISPLAY REFERENCE HISTOGRAM
# =========================================================

reference_histogram = calculate_histogram(
    reference
)


# =========================================================
# CREATE CLAHE
# =========================================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)


# =========================================================
# PROCESS ALL FOUR IMAGES
# =========================================================

for path, name in zip(
    image_paths,
    image_names
):

    # -----------------------------------------------------
    # Read image as grayscale
    # -----------------------------------------------------

    image = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        print(
            "Error: Could not read",
            path
        )

        continue


    # -----------------------------------------------------
    # Histogram Equalization
    # -----------------------------------------------------

    equalized_image = histogram_equalization(
        image
    )


    # -----------------------------------------------------
    # CLAHE
    # -----------------------------------------------------

    clahe_image = clahe.apply(
        image
    )


    # -----------------------------------------------------
    # Histogram Matching
    #
    # EVERY IMAGE IS MATCHED TO THE SAME
    # REFERENCE IMAGE
    # -----------------------------------------------------

    matched_image = histogram_matching(
        image,
        reference
    )


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

    matched_histogram = calculate_histogram(
        matched_image
    )


    # =====================================================
    # CREATE SEPARATE FIGURE FOR THIS IMAGE
    # =====================================================

    fig = plt.figure(
        figsize=(17, 10),
        num=f"Task 7 - {name}"
    )


    # =====================================================
    # TOP ROW - IMAGES
    # =====================================================

    # Original
    ax1 = fig.add_subplot(2, 4, 1)

    ax1.imshow(
        image,
        cmap="gray"
    )

    ax1.set_title(
        "Original",
        fontsize=13,
        fontweight="bold"
    )

    ax1.axis("off")


    # Histogram Equalization
    ax2 = fig.add_subplot(2, 4, 2)

    ax2.imshow(
        equalized_image,
        cmap="gray"
    )

    ax2.set_title(
        "Histogram Equalization",
        fontsize=13,
        fontweight="bold"
    )

    ax2.axis("off")


    # CLAHE
    ax3 = fig.add_subplot(2, 4, 3)

    ax3.imshow(
        clahe_image,
        cmap="gray"
    )

    ax3.set_title(
        "CLAHE",
        fontsize=13,
        fontweight="bold"
    )

    ax3.axis("off")


    # Histogram Matching
    ax4 = fig.add_subplot(2, 4, 4)

    ax4.imshow(
        matched_image,
        cmap="gray"
    )

    ax4.set_title(
        "Histogram Matching",
        fontsize=13,
        fontweight="bold"
    )

    ax4.axis("off")


    # =====================================================
    # BOTTOM ROW - HISTOGRAMS
    # =====================================================

    # Original Histogram
    ax5 = fig.add_subplot(2, 4, 5)

    ax5.bar(
        np.arange(256),
        original_histogram,
        width=1
    )

    ax5.set_title(
        "Original Histogram",
        fontweight="bold"
    )

    ax5.set_xlabel(
        "Pixel Intensity (0-255)"
    )

    ax5.set_ylabel(
        "Frequency"
    )

    ax5.set_xlim(
        0,
        255
    )


    # Equalized Histogram
    ax6 = fig.add_subplot(2, 4, 6)

    ax6.bar(
        np.arange(256),
        equalized_histogram,
        width=1
    )

    ax6.set_title(
        "Equalized Histogram",
        fontweight="bold"
    )

    ax6.set_xlabel(
        "Pixel Intensity (0-255)"
    )

    ax6.set_ylabel(
        "Frequency"
    )

    ax6.set_xlim(
        0,
        255
    )


    # CLAHE Histogram
    ax7 = fig.add_subplot(2, 4, 7)

    ax7.bar(
        np.arange(256),
        clahe_histogram,
        width=1
    )

    ax7.set_title(
        "CLAHE Histogram",
        fontweight="bold"
    )

    ax7.set_xlabel(
        "Pixel Intensity (0-255)"
    )

    ax7.set_ylabel(
        "Frequency"
    )

    ax7.set_xlim(
        0,
        255
    )


    # Matched Histogram
    ax8 = fig.add_subplot(2, 4, 8)

    ax8.bar(
        np.arange(256),
        matched_histogram,
        width=1
    )

    ax8.set_title(
        "Matched Histogram",
        fontweight="bold"
    )

    ax8.set_xlabel(
        "Pixel Intensity (0-255)"
    )

    ax8.set_ylabel(
        "Frequency"
    )

    ax8.set_xlim(
        0,
        255
    )


    # =====================================================
    # FIGURE TITLE
    # =====================================================

    fig.suptitle(
        f"Task 7: Final Comparison - {name}",
        fontsize=18,
        fontweight="bold"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.94]
    )


# =========================================================
# SHOW ALL FOUR FIGURES AT ONCE
# =========================================================

plt.show()