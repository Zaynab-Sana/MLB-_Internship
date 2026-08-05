
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
import edge_utils as eu


IMAGE_PATH = Path(__file__).resolve().parent.parent / "TASK1" / "image.jpg"

# Read grayscale image
image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Image not found at {IMAGE_PATH}!")
    exit()

# Add Gaussian noise so LoG's smoothing benefit is visible
noisy = eu.add_gaussian_noise(image, noise_percentage=5)

# Plain Laplacian (no smoothing) on the noisy image
laplacian_noisy = eu.laplacian_manual(noisy)

# Laplacian of Gaussian: Gaussian blur first, then Laplacian
blurred = eu.gaussian_blur_manual(noisy, size=5, sigma=1.4)
log_output = eu.laplacian_manual(blurred)

# Convert to uint8 for display - take absolute value since edges show up
# as the magnitude of intensity change, in either direction
laplacian_display = eu.to_uint8(np.abs(laplacian_noisy))
log_display = eu.to_uint8(np.abs(log_output))

# Display results
plt.figure(figsize=(16, 5))

plt.subplot(1, 4, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(noisy, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(laplacian_display, cmap="gray")
plt.title("Laplacian (no smoothing)\non noisy image")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(log_display, cmap="gray")
plt.title("Laplacian of Gaussian (LoG)\non noisy image")
plt.axis("off")

plt.tight_layout()

print("\n========== Task 4: Laplacian vs. LoG on Noisy Image ==========\n")
print("Plain Laplacian on a noisy image:")
print("- The Laplacian is a second-derivative operator, so it strongly")
print("  amplifies high-frequency noise along with real edges.")
print("- Result: a noisy, speckled edge map with many false edges from noise.\n")
print("Laplacian of Gaussian (LoG):")
print("- Gaussian smoothing is applied first, which averages out random")
print("  noise while preserving larger-scale intensity changes (edges).")
print("- The Laplacian is then computed on the smoothed image, so it")
print("  reacts far less to noise and produces a cleaner edge map.\n")
print("Conclusion:")
print("LoG is much more robust to noise than a plain Laplacian because the")
print("Gaussian pre-blur suppresses the very high-frequency content (noise)")
print("that a second-derivative operator is most sensitive to.")

plt.show()