import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load the two images

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


# Create SIFT detector

sift = cv2.SIFT_create()


# Detect keypoints and generate descriptors

keypoints1, descriptors1 = sift.detectAndCompute(
    gray1,
    None
)

keypoints2, descriptors2 = sift.detectAndCompute(
    gray2,
    None
)


print("Feature Detection")
print("------------------")

print("Image 1 keypoints:", len(keypoints1))
print("Image 2 keypoints:", len(keypoints2))


if descriptors1 is None or descriptors2 is None:
    print("Descriptors could not be generated.")
    exit()


# Create BF Matcher

bf = cv2.BFMatcher(
    cv2.NORM_L2,
    crossCheck=False
)


# KNN matching

knn_matches = bf.knnMatch(
    descriptors1,
    descriptors2,
    k=2
)


print("\nKNN Matching")
print("------------")

print("Total KNN matches:", len(knn_matches))


# Apply Lowe's ratio test

good_matches = []

ratio_threshold = 0.75

for match_pair in knn_matches:

    if len(match_pair) == 2:

        best_match, second_best_match = match_pair

        if best_match.distance < ratio_threshold * second_best_match.distance:

            good_matches.append(best_match)


print("\nRatio Test")
print("----------")

print("Ratio threshold:", ratio_threshold)
print("Good matches:", len(good_matches))


# Check whether enough matches are available for RANSAC

if len(good_matches) < 4:

    print(
        "\nNot enough good matches for RANSAC."
    )

    print(
        "At least 4 good matches are required."
    )

    exit()


# Get the coordinates of the matched keypoints

points1 = np.float32([
    keypoints1[m.queryIdx].pt
    for m in good_matches
]).reshape(-1, 1, 2)


points2 = np.float32([
    keypoints2[m.trainIdx].pt
    for m in good_matches
]).reshape(-1, 1, 2)


# Apply RANSAC

homography_matrix, mask = cv2.findHomography(
    points1,
    points2,
    cv2.RANSAC,
    5.0
)


if mask is None:

    print("\nRANSAC could not find a valid transformation.")
    exit()


# Convert RANSAC mask to a simple array

mask = mask.ravel()


# Separate inliers and outliers

inlier_matches = []

outlier_matches = []

for i, match in enumerate(good_matches):

    if mask[i] == 1:
        inlier_matches.append(match)

    else:
        outlier_matches.append(match)


# Print RANSAC results

print("\nRANSAC Results")
print("--------------")

print("Total good matches:", len(good_matches))

print("RANSAC inliers:", len(inlier_matches))

print("RANSAC outliers:", len(outlier_matches))


# Draw matches before RANSAC

before_ransac = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


# Draw matches after RANSAC

after_ransac = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    inlier_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


# Convert images from BGR to RGB

before_ransac_rgb = cv2.cvtColor(
    before_ransac,
    cv2.COLOR_BGR2RGB
)

after_ransac_rgb = cv2.cvtColor(
    after_ransac,
    cv2.COLOR_BGR2RGB
)


# Display before and after RANSAC

plt.figure(figsize=(18, 8))

plt.subplot(1, 2, 1)

plt.imshow(before_ransac_rgb)

plt.title(
    f"Good Matches Before RANSAC\n"
    f"Total Good Matches: {len(good_matches)}"
)

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(after_ransac_rgb)

plt.title(
    f"Matches After RANSAC\n"
    f"Inliers: {len(inlier_matches)}"
)

plt.axis("off")

plt.tight_layout()



# Print final observation

print("\nObservation")
print("-----------")

print(
    "RANSAC was used to remove incorrect feature matches "
    "from the good matches obtained after the ratio test."
)

print(
    "Matches that agree with a common geometric transformation "
    "are classified as inliers."
)

print(
    "Matches that do not agree with the transformation "
    "are classified as outliers."
)

print(
    "Therefore, RANSAC makes feature matching more reliable "
    "by removing geometrically inconsistent matches."
)

plt.show()
