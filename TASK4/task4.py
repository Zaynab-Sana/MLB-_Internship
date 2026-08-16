import cv2
import numpy as np
import matplotlib.pyplot as plt

# Task 4: Image Warping

# Step 1: Read the image
image = cv2.imread("image.jpg")

if image is None:
    print("Image not found!")
    exit()

# Convert BGR to RGB for displaying with Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

height, width = image.shape[:2]

print("Original Image Size:")
print("Width:", width)
print("Height:", height)


# Step 2: Create a simple homography matrix

H = np.array([
    [1.0, 0.2, 30.0],
    [0.1, 1.0, 20.0],
    [0.0005, 0.0008, 1.0]
], dtype=float)

print("\nHomography Matrix:")
print(H)


# Step 3: Find inverse of homography

H_inverse = np.linalg.inv(H)

print("\nInverse Homography Matrix:")
print(H_inverse)


# Step 4: Create output image

output_height = height
output_width = width

manual_warp = np.zeros_like(image_rgb)


# Step 5: Manual image warping

for y in range(output_height):

    for x in range(output_width):

        # Output pixel in homogeneous coordinates
        output_point = np.array([
            x,
            y,
            1
        ], dtype=float)

        # Map output point back to source image
        source_point = H_inverse @ output_point

        # Convert from homogeneous coordinates
        w = source_point[2]

        if w == 0:
            continue

        source_x = source_point[0] / w
        source_y = source_point[1] / w

        # Convert coordinates to integer
        source_x = int(round(source_x))
        source_y = int(round(source_y))

        # Check if source coordinate is inside image
        if (
            source_x >= 0
            and source_x < width
            and source_y >= 0
            and source_y < height
        ):

            # Copy pixel manually
            manual_warp[y, x] = image_rgb[
                source_y,
                source_x
            ]


# Step 6: OpenCV warpPerspective

opencv_warp = cv2.warpPerspective(
    image,
    H,
    (output_width, output_height)
)

# Convert OpenCV result to RGB
opencv_warp_rgb = cv2.cvtColor(
    opencv_warp,
    cv2.COLOR_BGR2RGB
)


# Step 7: Display results

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(manual_warp)
plt.title("Manual Warping")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(opencv_warp_rgb)
plt.title("OpenCV warpPerspective")
plt.axis("off")

plt.tight_layout()
plt.show()