import cv2
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


print("Feature Detection Results")
print("-------------------------")

print("Image 1 keypoints:", len(keypoints1))
print("Image 2 keypoints:", len(keypoints2))


if descriptors1 is None or descriptors2 is None:
    print("Could not generate descriptors.")
    exit()


print("Image 1 descriptor shape:", descriptors1.shape)
print("Image 2 descriptor shape:", descriptors2.shape)


# Create BF Matcher

bf = cv2.BFMatcher(
    cv2.NORM_L2,
    crossCheck=False
)


# Perform KNN matching

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


print("\nRatio Test Results")
print("------------------")

print("Ratio threshold:", ratio_threshold)

print("Good matches:", len(good_matches))


# Draw the good matches

matched_image = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


# Convert BGR image to RGB for Matplotlib

matched_image_rgb = cv2.cvtColor(
    matched_image,
    cv2.COLOR_BGR2RGB
)


# Display the original images

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)

plt.imshow(
    cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
)

plt.title(
    f"Image 1\nKeypoints: {len(keypoints1)}"
)

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
)

plt.title(
    f"Image 2\nKeypoints: {len(keypoints2)}"
)

plt.axis("off")

plt.tight_layout()


# Display the good matches

plt.figure(figsize=(16, 8))

plt.imshow(matched_image_rgb)

plt.title(
    f"Good Feature Matches Using SIFT + BF Matcher + KNN + Ratio Test\n"
    f"Good Matches: {len(good_matches)}"
)

plt.axis("off")

plt.tight_layout()
# Print observation

print("\nObservation")
print("-----------")

print(
    "SIFT was used to detect keypoints and generate descriptors "
    "in both images."
)

print(
    "BF Matcher was used to compare the SIFT descriptors "
    "using Euclidean distance."
)

print(
    "KNN matching with k=2 was used to find the two nearest "
    "candidate matches for each descriptor."
)

print(
    "Lowe's ratio test was applied to remove ambiguous matches."
)

print(
    "The remaining matches are considered good feature matches "
    "between the two images."
)

plt.show()


