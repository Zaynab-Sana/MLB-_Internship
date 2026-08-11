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

# Convert the grayscale image to float32 for Harris detection
gray_float = np.float32(gray)

# Define different threshold values
harris_thresholds = [0.01, 0.04, 0.08]
shi_tomasi_thresholds = [0.01, 0.04, 0.08]

# Store the number of detected corners
harris_counts = []
shi_tomasi_counts = []

# Harris Corner Detection

for threshold in harris_thresholds:

    # Calculate Harris corner response
    harris_response = cv2.cornerHarris(
        gray_float,
        blockSize=2,
        ksize=3,
        k=0.04
    )

    # Dilate the response to make the detected corners easier to see
    harris_response = cv2.dilate(harris_response, None)

    # Create a copy of the original image
    harris_image = image_rgb.copy()

    # Select corners according to the threshold
    corner_points = harris_response > (
        threshold * harris_response.max()
    )

    # Count detected corners
    number_of_corners = np.sum(corner_points)

    harris_counts.append(number_of_corners)

    # Mark detected corners
    harris_image[corner_points] = [255, 0, 0]

    # Create a figure
    plt.figure(figsize=(8, 6))

    plt.imshow(harris_image)
    plt.title(
        f"Harris Corner Detection\n"
        f"Threshold = {threshold} | "
        f"Corners = {number_of_corners}"
    )
    plt.axis("off")
    plt.tight_layout()


# Shi-Tomasi Corner Detection

for threshold in shi_tomasi_thresholds:

    # Detect candidate corners using Shi-Tomasi
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=1000,
        qualityLevel=threshold,
        minDistance=10,
        blockSize=3
    )

    # Create a copy of the original image
    shi_image = image_rgb.copy()

    # Count detected corners
    if corners is not None:
        number_of_corners = len(corners)

        # Convert corner coordinates to integers
        corners = np.int32(corners)

        # Draw each detected corner
        for corner in corners:
            x, y = corner.ravel()

            cv2.circle(
                shi_image,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )
    else:
        number_of_corners = 0

    shi_tomasi_counts.append(number_of_corners)

    # Create a figure
    plt.figure(figsize=(8, 6))

    plt.imshow(shi_image)
    plt.title(
        f"Shi-Tomasi Corner Detection\n"
        f"Threshold = {threshold} | "
        f"Corners = {number_of_corners}"
    )
    plt.axis("off")
    plt.tight_layout()


# Show all figures at the same time

# Print the comparison results

print("\nTask 2: Harris and Shi-Tomasi Corner Detection")
print("-" * 70)

print(
    f"{'Threshold':<15}"
    f"{'Harris Corners':<20}"
    f"{'Shi-Tomasi Corners':<20}"
)

print("-" * 70)

for i in range(len(harris_thresholds)):

    print(
        f"{harris_thresholds[i]:<15}"
        f"{harris_counts[i]:<20}"
        f"{shi_tomasi_counts[i]:<20}"
    )

print("-" * 70)


# Print the comparison observation

print("\nObservation:")
print(
    "Harris and Shi-Tomasi both detect corners by looking for "
    "strong intensity changes in two directions."
)

print(
    "When the threshold is increased, fewer but stronger corners "
    "are generally detected."
)

print(
    "Harris and Shi-Tomasi may detect different numbers and "
    "locations of corners because they use different corner "
    "response calculations."
)
plt.show()


