import numpy as np
import math

# Task 1: Homogeneous Coordinates

# Step 1: Define 2D points
points = np.array([
    [2, 3],
    [4, 3],
    [4, 5],
    [2, 5]
], dtype=float)

print("Original 2D Points:")
print(points)

# Step 2: Convert 2D points to Homogeneous Coordinates
homogeneous_points = np.ones((points.shape[0], 3))

homogeneous_points[:, 0] = points[:, 0]
homogeneous_points[:, 1] = points[:, 1]

print("\nHomogeneous Coordinates:")
print(homogeneous_points)

# Step 3: Translation Matrix
tx = 5
ty = 3

translation_matrix = np.array([
    [1, 0, tx],
    [0, 1, ty],
    [0, 0, 1]
], dtype=float)

print("\nTranslation Matrix:")
print(translation_matrix)

translated_points = (translation_matrix @ homogeneous_points.T).T

print("\nTranslated Points:")
print(translated_points)

# Step 4: Scaling Matrix
sx = 2
sy = 2

scaling_matrix = np.array([
    [sx, 0, 0],
    [0, sy, 0],
    [0, 0, 1]
], dtype=float)

print("\nScaling Matrix:")
print(scaling_matrix)

scaled_points = (scaling_matrix @ homogeneous_points.T).T

print("\nScaled Points:")
print(scaled_points)

# Step 5: Rotation Matrix
angle = 45

theta = angle * math.pi / 180

cos_theta = math.cos(theta)
sin_theta = math.sin(theta)

rotation_matrix = np.array([
    [cos_theta, -sin_theta, 0],
    [sin_theta, cos_theta, 0],
    [0, 0, 1]
], dtype=float)

print("\nRotation Matrix:")
print(rotation_matrix)

rotated_points = (rotation_matrix @ homogeneous_points.T).T

print("\nRotated Points:")
print(rotated_points)

# Step 6: Convert homogeneous coordinates back to 2D
final_points = rotated_points[:, :2] / rotated_points[:, 2:]

print("\nFinal 2D Points after Rotation:")
print(final_points)