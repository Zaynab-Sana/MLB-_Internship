import cv2
import numpy as np


# Create example floating-point descriptors for Euclidean distance

descriptor_1 = np.array(
    [1.0, 2.0, 3.0, 4.0],
    dtype=np.float32
)

descriptor_2 = np.array(
    [2.0, 4.0, 3.0, 7.0],
    dtype=np.float32
)

# Calculate Euclidean distance
euclidean_distance = np.linalg.norm(
    descriptor_1 - descriptor_2
)

print("Euclidean Distance Example")
print("--------------------------")

print("Descriptor 1:", descriptor_1)
print("Descriptor 2:", descriptor_2)

print("Euclidean Distance:",
      euclidean_distance)


# Create example binary descriptors for Hamming distance

binary_descriptor_1 = np.array(
    [[1, 0, 1, 1, 0, 1, 0, 0]],
    dtype=np.uint8
)

binary_descriptor_2 = np.array(
    [[1, 1, 0, 1, 0, 0, 0, 1]],
    dtype=np.uint8
)

# Calculate Hamming distance manually
hamming_distance = np.sum(
    binary_descriptor_1 != binary_descriptor_2
)

print("\nHamming Distance Example")
print("------------------------")

print("Binary Descriptor 1:",
      binary_descriptor_1[0])

print("Binary Descriptor 2:",
      binary_descriptor_2[0])

print("Hamming Distance:",
      hamming_distance)


# Create actual SIFT descriptors from an image

image = cv2.imread("image.jpg")

if image is None:
    print("\nCould not read image1.jpg")
    exit()

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

# Create SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and generate descriptors
sift_keypoints, sift_descriptors = sift.detectAndCompute(
    gray,
    None
)

print("\nSIFT Descriptor Test")
print("--------------------")

if sift_descriptors is not None and len(sift_descriptors) >= 2:

    sift_descriptor_1 = sift_descriptors[0]
    sift_descriptor_2 = sift_descriptors[1]

    # Calculate Euclidean distance between two SIFT descriptors
    sift_euclidean_distance = np.linalg.norm(
        sift_descriptor_1 - sift_descriptor_2
    )

    print("Number of SIFT keypoints:",
          len(sift_keypoints))

    print("SIFT descriptor size:",
          sift_descriptors.shape[1])

    print("Euclidean distance between")
    print("SIFT descriptor 1 and descriptor 2:",
          sift_euclidean_distance)

else:

    print("Not enough SIFT descriptors were detected.")


# Create actual ORB descriptors from the same image

orb = cv2.ORB_create()

# Detect keypoints and generate binary descriptors
orb_keypoints, orb_descriptors = orb.detectAndCompute(
    gray,
    None
)

print("\nORB Descriptor Test")
print("-------------------")

if orb_descriptors is not None and len(orb_descriptors) >= 2:

    orb_descriptor_1 = orb_descriptors[0]
    orb_descriptor_2 = orb_descriptors[1]

    # Calculate Hamming distance using OpenCV
    hamming_distance_opencv = cv2.norm(
        orb_descriptor_1,
        orb_descriptor_2,
        cv2.NORM_HAMMING
    )

    print("Number of ORB keypoints:",
          len(orb_keypoints))

    print("ORB descriptor size:",
          orb_descriptors.shape[1])

    print("Hamming distance between")
    print("ORB descriptor 1 and descriptor 2:",
          hamming_distance_opencv)

else:

    print("Not enough ORB descriptors were detected.")


# Compare several example descriptors

print("\nDistance Comparison Examples")
print("-----------------------------")

# Example floating-point descriptors
float_descriptors = [
    (
        np.array([1, 2, 3, 4], dtype=np.float32),
        np.array([1, 2, 3, 4], dtype=np.float32)
    ),
    (
        np.array([1, 2, 3, 4], dtype=np.float32),
        np.array([2, 3, 4, 5], dtype=np.float32)
    ),
    (
        np.array([1, 2, 3, 4], dtype=np.float32),
        np.array([10, 10, 10, 10], dtype=np.float32)
    )
]

for i, (d1, d2) in enumerate(float_descriptors, 1):

    distance = np.linalg.norm(d1 - d2)

    print(
        f"Example {i} Euclidean Distance: "
        f"{distance:.4f}"
    )


# Example binary descriptors
binary_descriptors = [
    (
        np.array([1, 0, 1, 1, 0, 1, 0, 0]),
        np.array([1, 0, 1, 1, 0, 1, 0, 0])
    ),
    (
        np.array([1, 0, 1, 1, 0, 1, 0, 0]),
        np.array([1, 1, 0, 1, 0, 0, 0, 1])
    ),
    (
        np.array([1, 1, 1, 1, 1, 1, 1, 1]),
        np.array([0, 0, 0, 0, 0, 0, 0, 0])
    )
]

for i, (d1, d2) in enumerate(binary_descriptors, 1):

    distance = np.sum(d1 != d2)

    print(
        f"Example {i} Hamming Distance: "
        f"{distance}"
    )


# Print final explanation

print("\nExplanation")
print("-----------")

print(
    "Euclidean distance measures the straight-line distance "
    "between numerical descriptor vectors."
)

print(
    "It is commonly used with floating-point descriptors such "
    "as SIFT."
)

print(
    "Hamming distance counts the number of different binary "
    "bits between two binary descriptors."
)

print(
    "It is used with binary descriptors such as ORB, BRIEF, "
    "and BRISK."
)

print(
    "A smaller distance generally means that two descriptors "
    "are more similar."
)