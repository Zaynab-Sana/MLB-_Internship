import cv2
import matplotlib.pyplot as plt
import os

# Select three images
image_paths = [
    "image1.jpg",
    "image2.jpg",
    "image3.jpg"
]

# Create SIFT detector
sift = cv2.SIFT_create()

# Store results for the summary
results = []

# Process each image
for image_path in image_paths:

    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue

    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read image: {image_path}")
        continue

    # Convert BGR image to RGB for displaying with Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to grayscale for feature detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect keypoints in the image
    keypoints = sift.detect(gray, None)

    # Draw the detected keypoints on the image
    keypoint_image = cv2.drawKeypoints(
        image_rgb,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Count the detected keypoints
    number_of_keypoints = len(keypoints)

    # Create an observation based on the number of detected keypoints
    if number_of_keypoints > 300:
        observation = (
            "Many distinctive features were detected. "
            "The image contains several textured or corner-like regions."
        )

    elif number_of_keypoints > 100:
        observation = (
            "A moderate number of keypoints were detected. "
            "The image contains some distinctive corners, edges, and textures."
        )

    else:
        observation = (
            "Fewer keypoints were detected. "
            "The image contains relatively fewer distinctive regions or textures."
        )

    # Store the result
    results.append([
        os.path.basename(image_path),
        number_of_keypoints,
        observation
    ])

    # Create a separate figure for each image
    plt.figure(figsize=(12, 5))

    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title(f"Original Image\n{os.path.basename(image_path)}")
    plt.axis("off")

    # Display the image with detected keypoints
    plt.subplot(1, 2, 2)
    plt.imshow(keypoint_image)
    plt.title(f"Detected Keypoints\nTotal Keypoints = {number_of_keypoints}")
    plt.axis("off")

    plt.tight_layout()

# Display all three figures at the same time
# Print the summary of detected keypoints
print("\nTask 1: Image Features and Keypoints Summary")
print("-" * 100)

print(f"{'Image':<20} {'Keypoints':<15} Observation")
print("-" * 100)

for result in results:
    image_name, count, observation = result
    print(f"{image_name:<20} {count:<15} {observation}")

# Print the general observation
print("\nGeneral Observation:")
print(
    "Keypoints are mainly detected around distinctive regions such as "
    "corners, textured areas, and strong intensity changes. "
    "Flat regions usually contain fewer useful keypoints because they "
    "have little intensity variation."
)

# Explain the three types of image regions
print("\nDifference Between Image Regions:")
print("Flat Region: Very little intensity change, so it usually provides fewer useful features.")
print("Edge: Strong intensity change in one direction and represents a boundary.")
print("Corner: Strong intensity change in two directions and is usually a strong feature.")
plt.show()
