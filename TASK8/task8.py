import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load the two input images

image1 = cv2.imread("image1.jpg")
image2 = cv2.imread("image2.jpg")

if image1 is None:
    print("Could not read image1.jpg")
    exit()

if image2 is None:
    print("Could not read image2.jpg")
    exit()


# Convert images to grayscale

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


# Feature Detection and Description using SIFT

sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(
    gray1,
    None
)

keypoints2, descriptors2 = sift.detectAndCompute(
    gray2,
    None
)


if descriptors1 is None or descriptors2 is None:
    print("Could not generate descriptors.")
    exit()


# Display detected keypoints

keypoints_image1 = cv2.drawKeypoints(
    image1,
    keypoints1,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

keypoints_image2 = cv2.drawKeypoints(
    image2,
    keypoints2,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)


# Print detected keypoints

print("Feature Detection and Description")
print("----------------------------------")

print("Image 1 keypoints:", len(keypoints1))
print("Image 2 keypoints:", len(keypoints2))

print("Image 1 descriptor shape:", descriptors1.shape)
print("Image 2 descriptor shape:", descriptors2.shape)


# Create Brute-Force Matcher

bf = cv2.BFMatcher(
    cv2.NORM_L2,
    crossCheck=False
)


# KNN Matching

knn_matches = bf.knnMatch(
    descriptors1,
    descriptors2,
    k=2
)


# Total matches

total_matches = len(knn_matches)

print("\nMatching")
print("--------")

print("Total matches:", total_matches)


# Apply Lowe's Ratio Test

ratio_threshold = 0.75

good_matches = []

for match_pair in knn_matches:

    if len(match_pair) == 2:

        best_match, second_best_match = match_pair

        if best_match.distance < (
            ratio_threshold * second_best_match.distance
        ):

            good_matches.append(best_match)


# Print good matches

print("\nRatio Test")
print("----------")

print("Ratio threshold:", ratio_threshold)
print("Good matches:", len(good_matches))


# Check whether enough matches are available for RANSAC

if len(good_matches) < 4:

    print("\nNot enough good matches for RANSAC.")
    print("At least 4 good matches are required.")

    # Display keypoints even if RANSAC cannot be applied

    plt.figure(figsize=(16, 7))

    plt.subplot(1, 2, 1)

    plt.imshow(
        cv2.cvtColor(
            keypoints_image1,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(
        f"Image 1 - {len(keypoints1)} Keypoints"
    )

    plt.axis("off")


    plt.subplot(1, 2, 2)

    plt.imshow(
        cv2.cvtColor(
            keypoints_image2,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(
        f"Image 2 - {len(keypoints2)} Keypoints"
    )

    plt.axis("off")

    plt.tight_layout()
    plt.show()

    exit()


# Get coordinates of matched keypoints

points1 = np.float32([
    keypoints1[match.queryIdx].pt
    for match in good_matches
]).reshape(-1, 1, 2)


points2 = np.float32([
    keypoints2[match.trainIdx].pt
    for match in good_matches
]).reshape(-1, 1, 2)


# Apply RANSAC

homography_matrix, ransac_mask = cv2.findHomography(
    points1,
    points2,
    cv2.RANSAC,
    5.0
)


if ransac_mask is None:

    print("\nRANSAC could not find a valid transformation.")
    exit()


# Convert mask into a one-dimensional array

ransac_mask = ransac_mask.ravel()


# Separate RANSAC inliers

inlier_matches = []

for i, match in enumerate(good_matches):

    if ransac_mask[i] == 1:
        inlier_matches.append(match)


# Count inliers

ransac_inliers = len(inlier_matches)


# Print RANSAC results

print("\nRANSAC")
print("------")

print("RANSAC inliers:", ransac_inliers)

print(
    "RANSAC outliers:",
    len(good_matches) - ransac_inliers
)


# Draw final matching visualization

final_matches_image = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    inlier_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


# Convert BGR to RGB for Matplotlib

keypoints_image1_rgb = cv2.cvtColor(
    keypoints_image1,
    cv2.COLOR_BGR2RGB
)

keypoints_image2_rgb = cv2.cvtColor(
    keypoints_image2,
    cv2.COLOR_BGR2RGB
)

final_matches_rgb = cv2.cvtColor(
    final_matches_image,
    cv2.COLOR_BGR2RGB
)


# Display detected keypoints

plt.figure(figsize=(16, 7))

plt.subplot(1, 2, 1)

plt.imshow(keypoints_image1_rgb)

plt.title(
    f"Image 1 - Detected Keypoints: {len(keypoints1)}"
)

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(keypoints_image2_rgb)

plt.title(
    f"Image 2 - Detected Keypoints: {len(keypoints2)}"
)

plt.axis("off")

plt.tight_layout()


# Display final matching result

plt.figure(figsize=(18, 8))

plt.imshow(final_matches_rgb)

plt.title(
    f"Final Feature Matching\n"
    f"Total Matches: {total_matches} | "
    f"Good Matches: {len(good_matches)} | "
    f"RANSAC Inliers: {ransac_inliers}"
)

plt.axis("off")

plt.tight_layout()
# Print final summary

print("\nFinal Results")
print("-------------")

print("Detected keypoints in Image 1:", len(keypoints1))
print("Detected keypoints in Image 2:", len(keypoints2))
print("Total matches:", total_matches)
print("Good matches after ratio test:", len(good_matches))
print("RANSAC inliers:", ransac_inliers)

print("\nPipeline completed successfully.")

print(
    "Feature Detection → Description → Matching → "
    "KNN → Ratio Test → RANSAC"
)

plt.show()


