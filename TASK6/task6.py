
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
import edge_utils as eu

IMAGE_PATH = Path(__file__).resolve().parent.parent / "TASK1" / "image.jpg"

image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Image not found at {IMAGE_PATH}!")
    exit()

# ---- Stage 1: Gaussian Smoothing ----
sigma = 1.4
kernel_size = 5
blurred = eu.gaussian_blur_manual(image, kernel_size, sigma)

# ---- Stage 2: Gradient Calculation ----
gx, gy, magnitude, direction = eu.sobel_gradients(blurred)

# ---- Stage 3: Non-Maximum Suppression ----
suppressed = eu.non_max_suppression(magnitude, direction)

# ---- Stage 4: Double Thresholding ----
low_ratio, high_ratio = 0.05, 0.15
thresholded, strong_val, weak_val = eu.double_threshold(suppressed, low_ratio, high_ratio)

# ---- Stage 5: Edge Tracking by Hysteresis ----
final_edges = eu.hysteresis(thresholded, weak_val, strong_val)

# Display every stage of the pipeline
plt.figure(figsize=(20, 6))

plt.subplot(1, 6, 1)
plt.imshow(image, cmap="gray")
plt.title("1. Original")
plt.axis("off")

plt.subplot(1, 6, 2)
plt.imshow(eu.to_uint8(blurred), cmap="gray")
plt.title("2. Gaussian\nSmoothing")
plt.axis("off")

plt.subplot(1, 6, 3)
plt.imshow(eu.to_uint8(magnitude), cmap="gray")
plt.title("3. Gradient\nMagnitude")
plt.axis("off")

plt.subplot(1, 6, 4)
plt.imshow(eu.to_uint8(suppressed), cmap="gray")
plt.title("4. Non-Max\nSuppression")
plt.axis("off")

plt.subplot(1, 6, 5)
plt.imshow(thresholded, cmap="gray")
plt.title("5. Double\nThreshold")
plt.axis("off")

plt.subplot(1, 6, 6)
plt.imshow(final_edges, cmap="gray")
plt.title("6. Final Edges\n(Hysteresis)")
plt.axis("off")

plt.tight_layout()

print("\n========== Task 6: Manual Canny Edge Detector ==========\n")
print("Pipeline stages:")
print(f"1. Gaussian Smoothing   - kernel size {kernel_size}, sigma {sigma}")
print("2. Gradient Calculation - manual Sobel Gx/Gy, magnitude & direction")
print("3. Non-Maximum Suppression - thins edges to 1px width along gradient direction")
print(f"4. Double Thresholding  - low ratio {low_ratio}, high ratio {high_ratio}")
print("   (both relative to the maximum gradient magnitude in the image)")
print("5. Edge Tracking by Hysteresis - weak edges connected to strong edges")
print("   are kept; isolated weak edges are discarded")
print(f"\nStrong-edge pixels found: {(thresholded == strong_val).sum()}")
print(f"Weak-edge pixels found before hysteresis: {(thresholded == weak_val).sum()}")
print(f"Final edge pixels after hysteresis: {(final_edges == strong_val).sum()}")

plt.show()