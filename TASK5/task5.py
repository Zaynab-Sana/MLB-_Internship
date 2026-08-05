
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
import edge_utils as eu

IMAGE_PATH = Path(__file__).resolve().parent.parent / "TASK1" / "image.jpg"

image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Image not found at {IMAGE_PATH}!")
    exit()

# Two sigma values - DoG approximates LoG well when sigma2 ~= 1.6 * sigma1
sigma1 = 1.0
sigma2 = 1.6
kernel_size = 9

blur1 = eu.gaussian_blur_manual(image, kernel_size, sigma1)
blur2 = eu.gaussian_blur_manual(image, kernel_size, sigma2)

dog = blur1 - blur2
dog_display = eu.to_uint8(np.abs(dog) * 4)  # scaled up for visibility

# True LoG for comparison (blur at an intermediate sigma, then Laplacian)
sigma_log = (sigma1 + sigma2) / 2
log_blurred = eu.gaussian_blur_manual(image, kernel_size, sigma_log)
log_output = eu.laplacian_manual(log_blurred)
log_display = eu.to_uint8(np.abs(log_output))

# Display the two blurs and the DoG result
plt.figure(figsize=(16, 5))

plt.subplot(1, 4, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(eu.to_uint8(blur1), cmap="gray")
plt.title(f"Gaussian Blur (sigma={sigma1})")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(eu.to_uint8(blur2), cmap="gray")
plt.title(f"Gaussian Blur (sigma={sigma2})")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(dog_display, cmap="gray")
plt.title(f"DoG (sigma {sigma1} - {sigma2})")
plt.axis("off")

plt.tight_layout()

# Separate figure: DoG vs. LoG side by side
plt.figure(figsize=(8, 5))

plt.subplot(1, 2, 1)
plt.imshow(dog_display, cmap="gray")
plt.title("Difference of Gaussian (DoG)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(log_display, cmap="gray")
plt.title("Laplacian of Gaussian (LoG)")
plt.axis("off")

plt.tight_layout()

print("\n========== Task 5: Difference of Gaussian (DoG) vs LoG ==========\n")
print(f"Two Gaussian blurs were computed with sigma={sigma1} and sigma={sigma2},")
print("then subtracted (blur1 - blur2) to produce the DoG output.\n")
print("Comparison with LoG:")
print("- DoG is a computationally cheap approximation of LoG: instead of")
print("  computing a second derivative, it uses the difference between two")
print("  differently-smoothed versions of the same image.")
print("- When sigma2 is chosen close to 1.6 x sigma1, DoG closely matches")
print("  the shape of a true LoG response, but only needs two blurs and a")
print("  subtraction - no explicit Laplacian kernel is required.")
print("- DoG is the basis of blob/keypoint detectors such as SIFT, precisely")
print("  because it's a fast, stable approximation of LoG's edge/blob response.")

plt.show()