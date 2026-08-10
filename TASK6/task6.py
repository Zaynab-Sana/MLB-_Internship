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
# FUNCTION: CALCULATE CDF
# =========================================================

def calculate_cdf(histogram):

    # Cumulative sum of histogram
    cdf = np.cumsum(histogram)

    # Normalize CDF between 0 and 1
    cdf = cdf / cdf[-1]

    return cdf


# =========================================================
# FUNCTION: HISTOGRAM MATCHING
# =========================================================

def histogram_matching(source, reference):

    # -----------------------------------------------------
    # Step 1: Calculate source histogram
    # -----------------------------------------------------

    source_histogram = calculate_histogram(source)


    # -----------------------------------------------------
    # Step 2: Calculate reference histogram
    # -----------------------------------------------------

    reference_histogram = calculate_histogram(reference)


    # -----------------------------------------------------
    # Step 3: Calculate source CDF
    # -----------------------------------------------------

    source_cdf = calculate_cdf(
        source_histogram
    )


    # -----------------------------------------------------
    # Step 4: Calculate reference CDF
    # -----------------------------------------------------

    reference_cdf = calculate_cdf(
        reference_histogram
    )


    # -----------------------------------------------------
    # Step 5: Create mapping
    # -----------------------------------------------------

    mapping = np.zeros(256, dtype=np.uint8)


    for source_intensity in range(256):

        # Difference between source CDF value
        # and every reference CDF value
        difference = np.abs(
            reference_cdf - source_cdf[source_intensity]
        )

        # Find reference intensity with
        # closest CDF value
        reference_intensity = np.argmin(
            difference
        )

        # Store mapping
        mapping[source_intensity] = (
            reference_intensity
        )


    # -----------------------------------------------------
    # Step 6: Apply mapping to source image
    # -----------------------------------------------------

    matched_image = mapping[source]

    return (
        matched_image,
        source_histogram,
        reference_histogram,
        source_cdf,
        reference_cdf
    )


# =========================================================
# READ SOURCE IMAGE
# =========================================================

source = cv2.imread(
    "source.png",
    cv2.IMREAD_GRAYSCALE
)


# =========================================================
# READ REFERENCE IMAGE
# =========================================================

reference = cv2.imread(
    "reference.jpg",
    cv2.IMREAD_GRAYSCALE
)


# =========================================================
# CHECK IMAGES
# =========================================================

if source is None:

    print("Error: Could not read source.jpg")
    exit()


if reference is None:

    print("Error: Could not read reference.jpg")
    exit()


# =========================================================
# PERFORM HISTOGRAM MATCHING
# =========================================================

(
    matched_image,
    source_histogram,
    reference_histogram,
    source_cdf,
    reference_cdf
) = histogram_matching(
    source,
    reference
)


# =========================================================
# CALCULATE MATCHED IMAGE HISTOGRAM
# =========================================================

matched_histogram = calculate_histogram(
    matched_image
)


# =========================================================
# DISPLAY RESULTS
# =========================================================

fig = plt.figure(
    figsize=(16, 10),
    num="Task 6 - Histogram Matching"
)


# =========================================================
# TOP ROW: IMAGES
# =========================================================


# ---------------------------------------------------------
# SOURCE IMAGE
# ---------------------------------------------------------

ax1 = fig.add_subplot(2, 3, 1)

ax1.imshow(
    source,
    cmap="gray"
)

ax1.set_title(
    "Source Image",
    fontsize=14,
    fontweight="bold"
)

ax1.axis("off")


# ---------------------------------------------------------
# REFERENCE IMAGE
# ---------------------------------------------------------

ax2 = fig.add_subplot(2, 3, 2)

ax2.imshow(
    reference,
    cmap="gray"
)

ax2.set_title(
    "Reference Image",
    fontsize=14,
    fontweight="bold"
)

ax2.axis("off")


# ---------------------------------------------------------
# MATCHED IMAGE
# ---------------------------------------------------------

ax3 = fig.add_subplot(2, 3, 3)

ax3.imshow(
    matched_image,
    cmap="gray"
)

ax3.set_title(
    "Matched Image",
    fontsize=14,
    fontweight="bold"
)

ax3.axis("off")


# =========================================================
# BOTTOM ROW: HISTOGRAMS
# =========================================================


# ---------------------------------------------------------
# SOURCE HISTOGRAM
# ---------------------------------------------------------

ax4 = fig.add_subplot(2, 3, 4)

ax4.bar(
    np.arange(256),
    source_histogram,
    width=1
)

ax4.set_title(
    "Source Histogram",
    fontsize=13,
    fontweight="bold"
)

ax4.set_xlabel(
    "Pixel Intensity (0-255)"
)

ax4.set_ylabel(
    "Frequency"
)

ax4.set_xlim(
    0,
    255
)


# ---------------------------------------------------------
# REFERENCE HISTOGRAM
# ---------------------------------------------------------

ax5 = fig.add_subplot(2, 3, 5)

ax5.bar(
    np.arange(256),
    reference_histogram,
    width=1
)

ax5.set_title(
    "Reference Histogram",
    fontsize=13,
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


# ---------------------------------------------------------
# MATCHED HISTOGRAM
# ---------------------------------------------------------

ax6 = fig.add_subplot(2, 3, 6)

ax6.bar(
    np.arange(256),
    matched_histogram,
    width=1
)

ax6.set_title(
    "Matched Histogram",
    fontsize=13,
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


# =========================================================
# MAIN TITLE
# =========================================================

fig.suptitle(
    "Task 6: Histogram Matching Using CDF",
    fontsize=18,
    fontweight="bold"
)


plt.tight_layout(
    rect=[0, 0, 1, 0.94]
)

plt.show()