import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# FUNCTION: Calculate Histogram Manually Using NumPy
# =========================================================

def calculate_histogram(image):

    histogram = np.zeros(256, dtype=int)

    for pixel in image.flatten():
        histogram[pixel] += 1

    return histogram


# =========================================================
# FUNCTION: Analyze Image
# =========================================================

def analyze_image(image):

    mean_intensity = np.mean(image)
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    std_intensity = np.std(image)

    # Brightness
    if mean_intensity < 85:
        brightness = "Dark"
    elif mean_intensity > 170:
        brightness = "Bright"
    else:
        brightness = "Normal"

    # Contrast
    if std_intensity < 40:
        contrast = "Low Contrast"
    else:
        contrast = "High Contrast"

    print(f"Mean Intensity : {mean_intensity:.2f}")
    print(f"Minimum Pixel  : {min_intensity}")
    print(f"Maximum Pixel  : {max_intensity}")
    print(f"Standard Dev.  : {std_intensity:.2f}")
    print(f"Brightness     : {brightness}")
    print(f"Contrast       : {contrast}")

    print("\nObservation:")

    if brightness == "Dark":
        print("- Most pixels have low intensity values.")
        print("- Histogram is concentrated toward the left side.")

    elif brightness == "Bright":
        print("- Most pixels have high intensity values.")
        print("- Histogram is concentrated toward the right side.")

    else:
        print("- Pixel intensities are mainly in the middle range.")

    if contrast == "Low Contrast":
        print("- Pixel intensities are concentrated in a narrow range.")
        print("- Histogram has a narrow spread.")

    else:
        print("- Pixel intensities are spread over a wider range.")
        print("- Histogram shows stronger contrast.")


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
# READ ALL IMAGES
# =========================================================

images = []
histograms = []

for path in image_paths:

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Could not read:", path)
        continue

    histogram = calculate_histogram(image)

    images.append(image)
    histograms.append(histogram)


# =========================================================
# CHECK THAT ALL FOUR IMAGES WERE LOADED
# =========================================================

if len(images) != 4:

    print("\nPlease make sure all four images exist:")
    print("1. bright.jpg")
    print("2. dark.jpg")
    print("3. highcontrast.jpg")
    print("4. lowcontrast.jpg")

else:

    # =====================================================
    # FIGURE 1
    # =====================================================

    fig1 = plt.figure(
        figsize=(14, 9),
        num="Figure 1 - Bright and Dark Images"
    )

    # Bright Image
    ax1 = fig1.add_subplot(2, 2, 1)

    ax1.imshow(images[0], cmap="gray")

    ax1.set_title(
        "Bright Image",
        fontsize=14,
        fontweight="bold"
    )

    ax1.axis("off")


    # Bright Histogram
    ax2 = fig1.add_subplot(2, 2, 2)

    ax2.bar(
        range(256),
        histograms[0],
        width=1
    )

    ax2.set_title(
        "Histogram - Bright Image",
        fontsize=14,
        fontweight="bold"
    )

    ax2.set_xlabel("Pixel Intensity (0-255)")
    ax2.set_ylabel("Frequency")
    ax2.set_xlim(0, 255)


    # Dark Image
    ax3 = fig1.add_subplot(2, 2, 3)

    ax3.imshow(images[1], cmap="gray")

    ax3.set_title(
        "Dark Image",
        fontsize=14,
        fontweight="bold"
    )

    ax3.axis("off")


    # Dark Histogram
    ax4 = fig1.add_subplot(2, 2, 4)

    ax4.bar(
        range(256),
        histograms[1],
        width=1
    )

    ax4.set_title(
        "Histogram - Dark Image",
        fontsize=14,
        fontweight="bold"
    )

    ax4.set_xlabel("Pixel Intensity (0-255)")
    ax4.set_ylabel("Frequency")
    ax4.set_xlim(0, 255)


    fig1.suptitle(
        "Bright and Dark Images with Histograms",
        fontsize=17,
        fontweight="bold"
    )

    fig1.tight_layout()


    # =====================================================
    # FIGURE 2
    # =====================================================

    fig2 = plt.figure(
        figsize=(14, 9),
        num="Figure 2 - Contrast Comparison"
    )

    # High Contrast Image
    ax5 = fig2.add_subplot(2, 2, 1)

    ax5.imshow(images[2], cmap="gray")

    ax5.set_title(
        "High Contrast Image",
        fontsize=14,
        fontweight="bold"
    )

    ax5.axis("off")


    # High Contrast Histogram
    ax6 = fig2.add_subplot(2, 2, 2)

    ax6.bar(
        range(256),
        histograms[2],
        width=1
    )

    ax6.set_title(
        "Histogram - High Contrast Image",
        fontsize=14,
        fontweight="bold"
    )

    ax6.set_xlabel("Pixel Intensity (0-255)")
    ax6.set_ylabel("Frequency")
    ax6.set_xlim(0, 255)


    # Low Contrast Image
    ax7 = fig2.add_subplot(2, 2, 3)

    ax7.imshow(images[3], cmap="gray")

    ax7.set_title(
        "Low Contrast Image",
        fontsize=14,
        fontweight="bold"
    )

    ax7.axis("off")


    # Low Contrast Histogram
    ax8 = fig2.add_subplot(2, 2, 4)

    ax8.bar(
        range(256),
        histograms[3],
        width=1
    )

    ax8.set_title(
        "Histogram - Low Contrast Image",
        fontsize=14,
        fontweight="bold"
    )

    ax8.set_xlabel("Pixel Intensity (0-255)")
    ax8.set_ylabel("Frequency")
    ax8.set_xlim(0, 255)


    fig2.suptitle(
        "High and Low Contrast Images with Histograms",
        fontsize=17,
        fontweight="bold"
    )

    fig2.tight_layout()


    # =====================================================
    # SHOW BOTH FIGURES
    # =====================================================
    # =====================================================
    # PRINT ANALYSIS
    # =====================================================

    for image, name in zip(images, image_names):
        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)

        analyze_image(image)

        print("=" * 60)
    plt.show()


