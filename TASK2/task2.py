import numpy as np
import matplotlib.pyplot as plt

# Task 2: Homography Matrix

# Step 1: Define original 2D points
points = np.array([
    [1, 1],
    [5, 1],
    [5, 5],
    [1, 5]
], dtype=float)

print("Original Points:")
print(points)

# Step 2: Convert points to homogeneous coordinates
homogeneous_points = np.ones((points.shape[0], 3))

homogeneous_points[:, 0] = points[:, 0]
homogeneous_points[:, 1] = points[:, 1]

print("\nHomogeneous Coordinates:")
print(homogeneous_points)

# Step 3: Create a 3x3 Homography Matrix
H = np.array([
    [1.0, 0.2, 1.0],
    [0.1, 1.0, 0.5],
    [0.001, 0.002, 1.0]
])

print("\nHomography Matrix:")
print(H)

# Step 4: Apply Homography Matrix
transformed_points = (H @ homogeneous_points.T).T

print("\nTransformed Homogeneous Points:")
print(transformed_points)

# Step 5: Convert back from homogeneous coordinates
transformed_2d = np.zeros((transformed_points.shape[0], 2))

for i in range(transformed_points.shape[0]):
    w = transformed_points[i, 2]

    transformed_2d[i, 0] = transformed_points[i, 0] / w
    transformed_2d[i, 1] = transformed_points[i, 1] / w

print("\nTransformed 2D Points:")
print(transformed_2d)

# Step 6: Display original and transformed points
print("\nPoint Transformation:")

for i in range(len(points)):
    print(
        "Original:",
        points[i],
        "-> Transformed:",
        transformed_2d[i]
    )

# Step 7: Visualize the transformation
plt.figure(figsize=(8, 6))

# Original points
original_x = points[:, 0]
original_y = points[:, 1]

# Transformed points
transformed_x = transformed_2d[:, 0]
transformed_y = transformed_2d[:, 1]

plt.scatter(
    original_x,
    original_y,
    label="Original Points"
)

plt.scatter(
    transformed_x,
    transformed_y,
    label="Transformed Points"
)

# Connect original points
for i in range(len(points)):
    j = (i + 1) % len(points)

    plt.plot(
        [points[i, 0], points[j, 0]],
        [points[i, 1], points[j, 1]]
    )

# Connect transformed points
for i in range(len(transformed_2d)):
    j = (i + 1) % len(transformed_2d)

    plt.plot(
        [transformed_2d[i, 0], transformed_2d[j, 0]],
        [transformed_2d[i, 1], transformed_2d[j, 1]]
    )

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Homography Transformation")

plt.legend()
plt.grid()

plt.show()