import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# FUNCTION: Calculate Grayscale Histogram
# =========================================================

def calculate_histogram(image):

    # Create array for intensity values 0-255
    histogram = np.zeros(256, dtype=int)

    # Count frequency of every intensity value
    for intensity in image.flatten():
        histogram[intensity] += 1

    return histogram


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
# READ IMAGES AND CALCULATE HISTOGRAMS
# =========================================================

images = []
histograms = []

for path in image_paths:

    # Read image as grayscale
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Could not read", path)
        continue

    # Calculate histogram manually
    histogram = calculate_histogram(image)

    images.append(image)
    histograms.append(histogram)


# =========================================================
# CHECK ALL FOUR IMAGES
# =========================================================

if len(images) != 4:

    print("\nError: Make sure these four images are present:")
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
        figsize=(14, 9),
        num="Figure 1 - Grayscale Histograms"
    )

    # -----------------------------------------------------
    # Bright Image
    # -----------------------------------------------------

    ax1 = fig1.add_subplot(2, 2, 1)

    ax1.imshow(
        images[0],
        cmap="gray"
    )

    ax1.set_title(
        "Bright Image",
        fontsize=14,
        fontweight="bold"
    )

    ax1.axis("off")


    # -----------------------------------------------------
    # Bright Histogram
    # -----------------------------------------------------

    ax2 = fig1.add_subplot(2, 2, 2)

    ax2.bar(
        range(256),
        histograms[0],
        width=1
    )

    ax2.set_title(
        "Grayscale Histogram - Bright Image",
        fontsize=14,
        fontweight="bold"
    )

    ax2.set_xlabel("Intensity Value (0-255)")
    ax2.set_ylabel("Frequency")
    ax2.set_xlim(0, 255)


    # -----------------------------------------------------
    # Dark Image
    # -----------------------------------------------------

    ax3 = fig1.add_subplot(2, 2, 3)

    ax3.imshow(
        images[1],
        cmap="gray"
    )

    ax3.set_title(
        "Dark Image",
        fontsize=14,
        fontweight="bold"
    )

    ax3.axis("off")


    # -----------------------------------------------------
    # Dark Histogram
    # -----------------------------------------------------

    ax4 = fig1.add_subplot(2, 2, 4)

    ax4.bar(
        range(256),
        histograms[1],
        width=1
    )

    ax4.set_title(
        "Grayscale Histogram - Dark Image",
        fontsize=14,
        fontweight="bold"
    )

    ax4.set_xlabel("Intensity Value (0-255)")
    ax4.set_ylabel("Frequency")
    ax4.set_xlim(0, 255)


    fig1.suptitle(
        "Task 2: Grayscale Histogram - Bright and Dark Images",
        fontsize=17,
        fontweight="bold"
    )

    fig1.tight_layout()


    # =====================================================
    # FIGURE 2
    # High Contrast + Low Contrast
    # =====================================================

    fig2 = plt.figure(
        figsize=(14, 9),
        num="Figure 2 - Grayscale Histograms"
    )

    # -----------------------------------------------------
    # High Contrast Image
    # -----------------------------------------------------

    ax5 = fig2.add_subplot(2, 2, 1)

    ax5.imshow(
        images[2],
        cmap="gray"
    )

    ax5.set_title(
        "High Contrast Image",
        fontsize=14,
        fontweight="bold"
    )

    ax5.axis("off")


    # -----------------------------------------------------
    # High Contrast Histogram
    # -----------------------------------------------------

    ax6 = fig2.add_subplot(2, 2, 2)

    ax6.bar(
        range(256),
        histograms[2],
        width=1
    )

    ax6.set_title(
        "Grayscale Histogram - High Contrast Image",
        fontsize=14,
        fontweight="bold"
    )

    ax6.set_xlabel("Intensity Value (0-255)")
    ax6.set_ylabel("Frequency")
    ax6.set_xlim(0, 255)


    # -----------------------------------------------------
    # Low Contrast Image
    # -----------------------------------------------------

    ax7 = fig2.add_subplot(2, 2, 3)

    ax7.imshow(
        images[3],
        cmap="gray"
    )

    ax7.set_title(
        "Low Contrast Image",
        fontsize=14,
        fontweight="bold"
    )

    ax7.axis("off")


    # -----------------------------------------------------
    # Low Contrast Histogram
    # -----------------------------------------------------

    ax8 = fig2.add_subplot(2, 2, 4)

    ax8.bar(
        range(256),
        histograms[3],
        width=1
    )

    ax8.set_title(
        "Grayscale Histogram - Low Contrast Image",
        fontsize=14,
        fontweight="bold"
    )

    ax8.set_xlabel("Intensity Value (0-255)")
    ax8.set_ylabel("Frequency")
    ax8.set_xlim(0, 255)


    fig2.suptitle(
        "Task 2: Grayscale Histogram - Contrast Comparison",
        fontsize=17,
        fontweight="bold"
    )

    fig2.tight_layout()


    # =====================================================
    # SHOW TWO SEPARATE FIGURES
    # =====================================================
    # =====================================================
    # PRINT FREQUENCY TABLES
    # =====================================================

    for image, histogram, name in zip(
            images,
            histograms,
            image_names
    ):

        print("\n")
        print("=" * 70)
        print(name)
        print("=" * 70)

        print("Intensity Value\tFrequency")
        print("-" * 70)

        for intensity in range(256):
            print(
                f"{intensity:3d}\t\t{histogram[intensity]}"
            )

        print("-" * 70)

        # Verify total pixels
        total_pixels = np.sum(histogram)

        print("Total Pixels :", total_pixels)
        print(
            "Image Pixels :",
            image.shape[0] * image.shape[1]
        )

        print("=" * 70)
    plt.show()

