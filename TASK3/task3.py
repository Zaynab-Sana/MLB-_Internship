import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "image.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Could not read the image.")
    exit()

# Convert the image to RGB for displaying with Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define different FAST threshold values
fast_thresholds = [10, 30, 50]

# Store the number of detected FAST keypoints
fast_counts = []

# Detect FAST keypoints using different thresholds
for threshold in fast_thresholds:

    # Create FAST detector with the current threshold
    fast = cv2.FastFeatureDetector_create(
        threshold=threshold,
        nonmaxSuppression=True
    )

    # Detect keypoints
    keypoints = fast.detect(gray, None)

    # Count the detected keypoints
    number_of_keypoints = len(keypoints)

    fast_counts.append(number_of_keypoints)

    # Draw the detected keypoints
    fast_image = cv2.drawKeypoints(
        image_rgb,
        keypoints,
        None,
        color=None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Create a separate figure
    plt.figure(figsize=(8, 6))

    plt.imshow(fast_image)
    plt.title(
        f"FAST Feature Detection\n"
        f"Threshold = {threshold} | "
        f"Keypoints = {number_of_keypoints}"
    )
    plt.axis("off")
    plt.tight_layout()


# Show all FAST figures at the same time
plt.show()


# Detect Harris corners for comparison

gray_float = np.float32(gray)

harris_response = cv2.cornerHarris(
    gray_float,
    blockSize=2,
    ksize=3,
    k=0.04
)

harris_response = cv2.dilate(harris_response, None)

# Use a fixed threshold for comparison
harris_threshold = 0.04

harris_image = image_rgb.copy()

harris_corners = harris_response > (
    harris_threshold * harris_response.max()
)

harris_count = np.sum(harris_corners)

harris_image[harris_corners] = [255, 0, 0]


# Detect Shi-Tomasi corners for comparison

shi_tomasi_quality = 0.04

shi_corners = cv2.goodFeaturesToTrack(
    gray,
    maxCorners=1000,
    qualityLevel=shi_tomasi_quality,
    minDistance=10,
    blockSize=3
)

shi_image = image_rgb.copy()

if shi_corners is not None:

    shi_corners = np.int32(shi_corners)

    shi_count = len(shi_corners)

    for corner in shi_corners:

        x, y = corner.ravel()

        cv2.circle(
            shi_image,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )

else:
    shi_count = 0


# Detect FAST using the middle threshold for comparison

comparison_fast = cv2.FastFeatureDetector_create(
    threshold=30,
    nonmaxSuppression=True
)

comparison_fast_keypoints = comparison_fast.detect(
    gray,
    None
)

comparison_fast_count = len(comparison_fast_keypoints)

fast_comparison_image = cv2.drawKeypoints(
    image_rgb,
    comparison_fast_keypoints,
    None,
    color=None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)


# Display Harris, Shi-Tomasi, and FAST comparison

plt.figure(figsize=(15, 5))

# Harris result
plt.subplot(1, 3, 1)
plt.imshow(harris_image)
plt.title(
    f"Harris\nCorners = {harris_count}"
)
plt.axis("off")

# Shi-Tomasi result
plt.subplot(1, 3, 2)
plt.imshow(shi_image)
plt.title(
    f"Shi-Tomasi\nCorners = {shi_count}"
)
plt.axis("off")

# FAST result
plt.subplot(1, 3, 3)
plt.imshow(fast_comparison_image)
plt.title(
    f"FAST\nKeypoints = {comparison_fast_count}"
)
plt.axis("off")

plt.tight_layout()

# Print FAST threshold comparison

print("\nTask 3: FAST Feature Detector")
print("-" * 60)

print(f"{'FAST Threshold':<20}{'Keypoints Detected':<20}")
print("-" * 60)

for i in range(len(fast_thresholds)):

    print(
        f"{fast_thresholds[i]:<20}"
        f"{fast_counts[i]:<20}"
    )

print("-" * 60)


# Print detector comparison

print("\nComparison of FAST, Harris, and Shi-Tomasi")
print("-" * 60)

print(f"{'Detector':<20}{'Detected Points':<20}")
print("-" * 60)

print(f"{'Harris':<20}{harris_count:<20}")
print(f"{'Shi-Tomasi':<20}{shi_count:<20}")
print(f"{'FAST':<20}{comparison_fast_count:<20}")

print("-" * 60)


# Print observations

print("\nObservation:")

print(
    "FAST detects keypoints using a fast intensity comparison test "
    "around each candidate pixel."
)

print(
    "When the FAST threshold is increased, fewer and stronger "
    "keypoints are generally detected."
)

print(
    "Harris and Shi-Tomasi are corner detectors, while FAST is a "
    "faster corner/keypoint detector designed for efficient detection."
)

print(
    "The number and locations of detected points can differ because "
    "the three methods use different detection techniques.")

plt.show()

