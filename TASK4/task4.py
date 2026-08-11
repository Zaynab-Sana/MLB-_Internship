import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

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

# Store the comparison results
results = []

# Store the visualized results
visual_results = []


# SIFT detects keypoints and generates descriptors

sift = cv2.SIFT_create()

start_time = time.perf_counter()

sift_keypoints, sift_descriptors = sift.detectAndCompute(
    gray,
    None
)

sift_time = time.perf_counter() - start_time

sift_keypoint_count = len(sift_keypoints)

if sift_descriptors is not None:
    sift_descriptor_size = sift_descriptors.shape[1]
    sift_descriptor_type = "Float"
else:
    sift_descriptor_size = 0
    sift_descriptor_type = "Float"

sift_image = cv2.drawKeypoints(
    image_rgb,
    sift_keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

visual_results.append(
    ("SIFT", sift_image, sift_keypoint_count)
)

results.append([
    "SIFT",
    sift_keypoint_count,
    sift_descriptor_size,
    sift_descriptor_type,
    sift_time
])


# ORB detects keypoints and generates descriptors

orb = cv2.ORB_create()

start_time = time.perf_counter()

orb_keypoints, orb_descriptors = orb.detectAndCompute(
    gray,
    None
)

orb_time = time.perf_counter() - start_time

orb_keypoint_count = len(orb_keypoints)

if orb_descriptors is not None:
    orb_descriptor_size = orb_descriptors.shape[1]
    orb_descriptor_type = "Binary"
else:
    orb_descriptor_size = 0
    orb_descriptor_type = "Binary"

orb_image = cv2.drawKeypoints(
    image_rgb,
    orb_keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

visual_results.append(
    ("ORB", orb_image, orb_keypoint_count)
)

results.append([
    "ORB",
    orb_keypoint_count,
    orb_descriptor_size,
    orb_descriptor_type,
    orb_time
])


# BRISK detects keypoints and generates descriptors

brisk = cv2.BRISK_create()

start_time = time.perf_counter()

brisk_keypoints, brisk_descriptors = brisk.detectAndCompute(
    gray,
    None
)

brisk_time = time.perf_counter() - start_time

brisk_keypoint_count = len(brisk_keypoints)

if brisk_descriptors is not None:
    brisk_descriptor_size = brisk_descriptors.shape[1]
    brisk_descriptor_type = "Binary"
else:
    brisk_descriptor_size = 0
    brisk_descriptor_type = "Binary"

brisk_image = cv2.drawKeypoints(
    image_rgb,
    brisk_keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

visual_results.append(
    ("BRISK", brisk_image, brisk_keypoint_count)
)

results.append([
    "BRISK",
    brisk_keypoint_count,
    brisk_descriptor_size,
    brisk_descriptor_type,
    brisk_time
])


# BRIEF is a descriptor, so FAST is used to detect its keypoints

fast = cv2.FastFeatureDetector_create(
    threshold=30,
    nonmaxSuppression=True
)

brief_keypoints = fast.detect(gray, None)

# Check whether the OpenCV installation contains BRIEF
if hasattr(cv2, "xfeatures2d"):

    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    start_time = time.perf_counter()

    brief_keypoints, brief_descriptors = brief.compute(
        gray,
        brief_keypoints
    )

    brief_time = time.perf_counter() - start_time

    if brief_keypoints is None:
        brief_keypoints = []

    brief_keypoint_count = len(brief_keypoints)

    if brief_descriptors is not None:
        brief_descriptor_size = brief_descriptors.shape[1]
    else:
        brief_descriptor_size = 0

    brief_descriptor_type = "Binary"

    brief_image = cv2.drawKeypoints(
        image_rgb,
        brief_keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    visual_results.append(
        ("BRIEF", brief_image, brief_keypoint_count)
    )

    results.append([
        "BRIEF",
        brief_keypoint_count,
        brief_descriptor_size,
        brief_descriptor_type,
        brief_time
    ])

else:
    print(
        "BRIEF is not available in this OpenCV installation. "
        "Install opencv-contrib-python to use BRIEF."
    )


# Display the detected keypoints for all methods

for name, result_image, keypoint_count in visual_results:

    plt.figure(figsize=(9, 6))

    plt.imshow(result_image)

    plt.title(
        f"{name} Keypoints\n"
        f"Number of Keypoints = {keypoint_count}"
    )

    plt.axis("off")
    plt.tight_layout()

# Show all figures together
# Print the comparison table

print("\nTask 4: Feature Descriptor Comparison")
print("-" * 100)

print(
    f"{'Method':<12}"
    f"{'Keypoints':<15}"
    f"{'Descriptor Size':<18}"
    f"{'Descriptor Type':<20}"
    f"{'Speed (seconds)':<15}"
)

print("-" * 100)

for result in results:

    method = result[0]
    keypoints = result[1]
    descriptor_size = result[2]
    descriptor_type = result[3]
    speed = result[4]

    print(
        f"{method:<12}"
        f"{keypoints:<15}"
        f"{descriptor_size:<18}"
        f"{descriptor_type:<20}"
        f"{speed:<15.6f}"
    )

print("-" * 100)


# Print observations

print("\nObservation:")

print(
    "SIFT generally produces floating-point descriptors and is "
    "more computationally expensive than binary methods."
)

print(
    "ORB produces binary descriptors and is designed for fast "
    "feature detection and description."
)

print(
    "BRIEF produces binary descriptors and is computationally simple. "
    "It requires another detector, such as FAST, to provide keypoints."
)

print(
    "BRISK produces binary descriptors and provides both keypoint "
    "detection and description."
)

print(
    "The exact number of keypoints and processing time depend on "
    "the input image and the computer being used."
)
plt.show()


