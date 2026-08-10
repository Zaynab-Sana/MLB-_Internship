import cv2
import numpy as np
import matplotlib.pyplot as plt




# =========================================================
# FUNCTION: Calculate Histogram
# =========================================================

def calculate_histogram(image):

    histogram = np.zeros(256, dtype=int)

    for pixel in image.flatten():
        histogram[pixel] += 1

    return histogram


# =========================================================
# FUNCTION: Histogram Equalization Using CDF
# =========================================================

def histogram_equalization(image):

    # Step 1: Calculate histogram
    histogram = calculate_histogram(image)

    # Step 2: Calculate CDF
    cdf = np.cumsum(histogram)

    # Step 3: Find first non-zero CDF value
    cdf_min = cdf[cdf > 0][0]

    # Step 4: Total number of pixels
    total_pixels = image.size

    # Step 5: Apply histogram equalization formula
    equalized_values = (
        (cdf - cdf_min)
        / (total_pixels - cdf_min)
        * 255
    )

    # Step 6: Convert values to uint8
    equalized_values = np.round(
        equalized_values
    ).astype(np.uint8)

    # Step 7: Map original pixels to equalized values
    equalized_image = equalized_values[image]

    return equalized_image


# =========================================================
# IMAGE PATHS
# =========================================================

image_paths = [
    "bright.jpg",
    "dark.jpg",
    "highcontrast.jpg",
    "lowcontrast.jpg"
]

image_names = [
    "Bright Image",
    "Dark Image",
    "High Contrast Image",
    "Low Contrast Image"
]


# =========================================================
# READ IMAGES
# =========================================================

original_images = []
equalized_images = []

original_histograms = []
equalized_histograms = []


for path in image_paths:

    # Read as grayscale
    image = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        print("Error: Could not read", path)
        continue

    # Perform manual histogram equalization
    equalized_image = histogram_equalization(image)

    # Calculate histograms
    original_histogram = calculate_histogram(image)

    equalized_histogram = calculate_histogram(
        equalized_image
    )

    # Store results
    original_images.append(image)
    equalized_images.append(equalized_image)

    original_histograms.append(
        original_histogram
    )

    equalized_histograms.append(
        equalized_histogram
    )


# =========================================================
# CHECK ALL FOUR IMAGES
# =========================================================

if len(original_images) != 4:

    print("\nError: Make sure these four images exist:")
    print("1. bright.jpg")
    print("2. dark.jpg")
    print("3. highcontrast.jpg")
    print("4. lowcontrast.jpg")

else:

    # =====================================================
    # FIGURE 1
    # Bright + Dark
    # =====================================================

    fig1 = plt.figure(
        figsize=(16, 10),
        num="Figure 1 - Histogram Equalization"
    )


    # -----------------------------------------------------
    # BRIGHT IMAGE - ORIGINAL
    # -----------------------------------------------------

    ax1 = fig1.add_subplot(2, 4, 1)

    ax1.imshow(
        original_images[0],
        cmap="gray"
    )

    ax1.set_title(
        "Bright - Original",
        fontweight="bold"
    )

    ax1.axis("off")


    # -----------------------------------------------------
    # BRIGHT IMAGE - EQUALIZED
    # -----------------------------------------------------

    ax2 = fig1.add_subplot(2, 4, 2)

    ax2.imshow(
        equalized_images[0],
        cmap="gray"
    )

    ax2.set_title(
        "Bright - Equalized",
        fontweight="bold"
    )

    ax2.axis("off")


    # -----------------------------------------------------
    # BRIGHT IMAGE - ORIGINAL HISTOGRAM
    # -----------------------------------------------------

    ax3 = fig1.add_subplot(2, 4, 3)

    ax3.bar(
        range(256),
        original_histograms[0],
        width=1
    )

    ax3.set_title(
        "Original Histogram",
        fontweight="bold"
    )

    ax3.set_xlabel("Intensity")
    ax3.set_ylabel("Frequency")
    ax3.set_xlim(0, 255)


    # -----------------------------------------------------
    # BRIGHT IMAGE - EQUALIZED HISTOGRAM
    # -----------------------------------------------------

    ax4 = fig1.add_subplot(2, 4, 4)

    ax4.bar(
        range(256),
        equalized_histograms[0],
        width=1
    )

    ax4.set_title(
        "Equalized Histogram",
        fontweight="bold"
    )

    ax4.set_xlabel("Intensity")
    ax4.set_ylabel("Frequency")
    ax4.set_xlim(0, 255)


    # -----------------------------------------------------
    # DARK IMAGE - ORIGINAL
    # -----------------------------------------------------

    ax5 = fig1.add_subplot(2, 4, 5)

    ax5.imshow(
        original_images[1],
        cmap="gray"
    )

    ax5.set_title(
        "Dark - Original",
        fontweight="bold"
    )

    ax5.axis("off")


    # -----------------------------------------------------
    # DARK IMAGE - EQUALIZED
    # -----------------------------------------------------

    ax6 = fig1.add_subplot(2, 4, 6)

    ax6.imshow(
        equalized_images[1],
        cmap="gray"
    )

    ax6.set_title(
        "Dark - Equalized",
        fontweight="bold"
    )

    ax6.axis("off")


    # -----------------------------------------------------
    # DARK IMAGE - ORIGINAL HISTOGRAM
    # -----------------------------------------------------

    ax7 = fig1.add_subplot(2, 4, 7)

    ax7.bar(
        range(256),
        original_histograms[1],
        width=1
    )

    ax7.set_title(
        "Original Histogram",
        fontweight="bold"
    )

    ax7.set_xlabel("Intensity")
    ax7.set_ylabel("Frequency")
    ax7.set_xlim(0, 255)


    # -----------------------------------------------------
    # DARK IMAGE - EQUALIZED HISTOGRAM
    # -----------------------------------------------------

    ax8 = fig1.add_subplot(2, 4, 8)

    ax8.bar(
        range(256),
        equalized_histograms[1],
        width=1
    )

    ax8.set_title(
        "Equalized Histogram",
        fontweight="bold"
    )

    ax8.set_xlabel("Intensity")
    ax8.set_ylabel("Frequency")
    ax8.set_xlim(0, 255)


    fig1.suptitle(
        "Task 4: Histogram Equalization - Bright and Dark Images",
        fontsize=17,
        fontweight="bold"
    )

    fig1.tight_layout()


    # =====================================================
    # FIGURE 2
    # High Contrast + Low Contrast
    # =====================================================

    fig2 = plt.figure(
        figsize=(16, 10),
        num="Figure 2 - Histogram Equalization"
    )


    # -----------------------------------------------------
    # HIGH CONTRAST - ORIGINAL
    # -----------------------------------------------------

    ax9 = fig2.add_subplot(2, 4, 1)

    ax9.imshow(
        original_images[2],
        cmap="gray"
    )

    ax9.set_title(
        "High Contrast - Original",
        fontweight="bold"
    )

    ax9.axis("off")


    # -----------------------------------------------------
    # HIGH CONTRAST - EQUALIZED
    # -----------------------------------------------------

    ax10 = fig2.add_subplot(2, 4, 2)

    ax10.imshow(
        equalized_images[2],
        cmap="gray"
    )

    ax10.set_title(
        "High Contrast - Equalized",
        fontweight="bold"
    )

    ax10.axis("off")


    # -----------------------------------------------------
    # HIGH CONTRAST - ORIGINAL HISTOGRAM
    # -----------------------------------------------------

    ax11 = fig2.add_subplot(2, 4, 3)

    ax11.bar(
        range(256),
        original_histograms[2],
        width=1
    )

    ax11.set_title(
        "Original Histogram",
        fontweight="bold"
    )

    ax11.set_xlabel("Intensity")
    ax11.set_ylabel("Frequency")
    ax11.set_xlim(0, 255)


    # -----------------------------------------------------
    # HIGH CONTRAST - EQUALIZED HISTOGRAM
    # -----------------------------------------------------

    ax12 = fig2.add_subplot(2, 4, 4)

    ax12.bar(
        range(256),
        equalized_histograms[2],
        width=1
    )

    ax12.set_title(
        "Equalized Histogram",
        fontweight="bold"
    )

    ax12.set_xlabel("Intensity")
    ax12.set_ylabel("Frequency")
    ax12.set_xlim(0, 255)


    # -----------------------------------------------------
    # LOW CONTRAST - ORIGINAL
    # -----------------------------------------------------

    ax13 = fig2.add_subplot(2, 4, 5)

    ax13.imshow(
        original_images[3],
        cmap="gray"
    )

    ax13.set_title(
        "Low Contrast - Original",
        fontweight="bold"
    )

    ax13.axis("off")


    # -----------------------------------------------------
    # LOW CONTRAST - EQUALIZED
    # -----------------------------------------------------

    ax14 = fig2.add_subplot(2, 4, 6)

    ax14.imshow(
        equalized_images[3],
        cmap="gray"
    )

    ax14.set_title(
        "Low Contrast - Equalized",
        fontweight="bold"
    )

    ax14.axis("off")


    # -----------------------------------------------------
    # LOW CONTRAST - ORIGINAL HISTOGRAM
    # -----------------------------------------------------

    ax15 = fig2.add_subplot(2, 4, 7)

    ax15.bar(
        range(256),
        original_histograms[3],
        width=1
    )

    ax15.set_title(
        "Original Histogram",
        fontweight="bold"
    )

    ax15.set_xlabel("Intensity")
    ax15.set_ylabel("Frequency")
    ax15.set_xlim(0, 255)


    # -----------------------------------------------------
    # LOW CONTRAST - EQUALIZED HISTOGRAM
    # -----------------------------------------------------

    ax16 = fig2.add_subplot(2, 4, 8)

    ax16.bar(
        range(256),
        equalized_histograms[3],
        width=1
    )

    ax16.set_title(
        "Equalized Histogram",
        fontweight="bold"
    )

    ax16.set_xlabel("Intensity")
    ax16.set_ylabel("Frequency")
    ax16.set_xlim(0, 255)


    fig2.suptitle(
        "Task 4: Histogram Equalization - Contrast Comparison",
        fontsize=17,
        fontweight="bold"
    )

    fig2.tight_layout()


    # =====================================================
    # SHOW TWO SEPARATE FIGURES
    # =====================================================

    plt.show()