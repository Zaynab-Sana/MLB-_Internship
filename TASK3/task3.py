import numpy as np
import matplotlib.pyplot as plt

# Task 3: Homography Estimation using DLT and RANSAC

# Step 1: Define corresponding points
# Source points
src_points = np.array([
    [100, 100],
    [300, 100],
    [300, 300],
    [100, 300],
    [150, 150],
    [250, 150]
], dtype=float)

# Destination points
dst_points = np.array([
    [120, 110],
    [320, 130],
    [290, 310],
    [90, 280],
    [160, 160],
    [260, 170]
], dtype=float)

print("Source Points:")
print(src_points)

print("\nDestination Points:")
print(dst_points)


# Step 2: DLT Homography Estimation

def calculate_homography(src, dst):

    A = []

    for i in range(len(src)):

        x = src[i, 0]
        y = src[i, 1]

        u = dst[i, 0]
        v = dst[i, 1]

        row1 = [
            -x, -y, -1,
            0, 0, 0,
            u * x, u * y, u
        ]

        row2 = [
            0, 0, 0,
            -x, -y, -1,
            v * x, v * y, v
        ]

        A.append(row1)
        A.append(row2)

    A = np.array(A, dtype=float)

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(A)

    # Last row of V transpose gives the solution
    H = Vt[-1].reshape(3, 3)

    # Normalize H
    H = H / H[2, 2]

    return H


# Step 3: Calculate Homography
H = calculate_homography(src_points, dst_points)

print("\nEstimated Homography Matrix:")
print(H)


# Step 4: Apply Homography to points

def transform_points(points, H):

    transformed = []

    for point in points:

        x = point[0]
        y = point[1]

        homogeneous_point = np.array([
            x,
            y,
            1
        ])

        result = H @ homogeneous_point

        w = result[2]

        new_x = result[0] / w
        new_y = result[1] / w

        transformed.append([
            new_x,
            new_y
        ])

    return np.array(transformed)


predicted_points = transform_points(src_points, H)

print("\nPredicted Destination Points:")
print(predicted_points)


# Step 5: Calculate Reprojection Error

errors = []

for i in range(len(dst_points)):

    dx = predicted_points[i, 0] - dst_points[i, 0]
    dy = predicted_points[i, 1] - dst_points[i, 1]

    error = np.sqrt(dx * dx + dy * dy)

    errors.append(error)

errors = np.array(errors)

print("\nReprojection Errors:")
print(errors)


# Step 6: RANSAC

def ransac_homography(src, dst, iterations=1000, threshold=5):

    best_H = None
    best_inliers = []

    number_of_points = len(src)

    for iteration in range(iterations):

        # Randomly select 4 points
        indices = np.random.choice(
            number_of_points,
            4,
            replace=False
        )

        sample_src = src[indices]
        sample_dst = dst[indices]

        # Calculate homography using the 4 points
        H = calculate_homography(
            sample_src,
            sample_dst
        )

        # Transform all source points
        predicted = transform_points(src, H)

        current_inliers = []

        # Check error for every point
        for i in range(number_of_points):

            dx = predicted[i, 0] - dst[i, 0]
            dy = predicted[i, 1] - dst[i, 1]

            error = np.sqrt(
                dx * dx + dy * dy
            )

            if error < threshold:
                current_inliers.append(i)

        # Keep the model with the most inliers
        if len(current_inliers) > len(best_inliers):

            best_inliers = current_inliers
            best_H = H

    return best_H, best_inliers


# Step 7: Run RANSAC

best_H, inliers = ransac_homography(
    src_points,
    dst_points,
    iterations=1000,
    threshold=5
)

print("\nRANSAC Homography:")
print(best_H)

print("\nInlier Indices:")
print(inliers)

print("\nNumber of Inliers:")
print(len(inliers))


# Step 8: Visualize the points

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)

plt.scatter(
    src_points[:, 0],
    src_points[:, 1],
    label="Source Points"
)

for i in range(len(src_points)):
    plt.text(
        src_points[i, 0],
        src_points[i, 1],
        str(i)
    )

plt.title("Source Points")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.legend()


plt.subplot(1, 2, 2)

plt.scatter(
    dst_points[:, 0],
    dst_points[:, 1],
    label="Destination Points"
)

for i in range(len(dst_points)):

    if i in inliers:
        plt.text(
            dst_points[i, 0],
            dst_points[i, 1],
            str(i)
        )

plt.title("Destination Points with RANSAC Inliers")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.legend()

plt.show()