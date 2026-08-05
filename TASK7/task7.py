

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

# ---- Sobel ----
_, _, sobel_magnitude, _ = eu.sobel_gradients(image)
sobel_display = eu.to_uint8(sobel_magnitude)

# ---- Laplacian ----
laplacian_output = eu.laplacian_manual(image)
laplacian_display = eu.to_uint8(np.abs(laplacian_output))

# ---- LoG ----
log_blurred = eu.gaussian_blur_manual(image, size=5, sigma=1.4)
log_output = eu.laplacian_manual(log_blurred)
log_display = eu.to_uint8(np.abs(log_output))

# ---- DoG ----
blur1 = eu.gaussian_blur_manual(image, size=9, sigma=1.0)
blur2 = eu.gaussian_blur_manual(image, size=9, sigma=1.6)
dog_output = blur1 - blur2
dog_display = eu.to_uint8(np.abs(dog_output) * 4)

# ---- Canny ----
canny_output = eu.canny_manual(image, sigma=1.4, kernel_size=5, low_ratio=0.05, high_ratio=0.15)

# Display all results together
plt.figure(figsize=(20, 8))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(sobel_display, cmap="gray")
plt.title("Sobel")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(laplacian_display, cmap="gray")
plt.title("Laplacian")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(log_display, cmap="gray")
plt.title("Laplacian of Gaussian (LoG)")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(dog_display, cmap="gray")
plt.title("Difference of Gaussian (DoG)")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(canny_output, cmap="gray")
plt.title("Canny")
plt.axis("off")

plt.tight_layout()

print("\n================ Task 7: Edge Detection Comparison ================\n")

print("Sobel Operator")
print("- Strengths: simple, fast, gives both gradient magnitude and direction;")
print("  good at detecting strong, well-defined edges.")
print("- Limitations: sensitive to noise (first-derivative operator); edges")
print("  can be thick/blurred without extra thinning.")
print("- Use cases: quick edge previews, feeding gradient direction into")
print("  later stages (Canny itself uses Sobel internally).\n")

print("Laplacian")
print("- Strengths: single kernel, isotropic (responds equally to edges in")
print("  every direction), useful for detecting fine detail and blobs.")
print("- Limitations: very sensitive to noise (second derivative amplifies")
print("  high-frequency content); produces double edges around each boundary.")
print("- Use cases: sharpening (the basis of unsharp masking), detecting")
print("  isolated points/blobs in relatively clean images.\n")

print("Laplacian of Gaussian (LoG)")
print("- Strengths: Gaussian pre-smoothing suppresses noise before the")
print("  Laplacian is applied, giving much cleaner edges than plain Laplacian.")
print("- Limitations: more expensive to compute; choice of sigma trades off")
print("  noise suppression against how precisely edges are localized.")
print("- Use cases: edge/blob detection on noisy or real-world images,")
print("  scale-space feature detection.\n")

print("Difference of Gaussian (DoG)")
print("- Strengths: cheap, fast approximation of LoG using only two blurs")
print("  and a subtraction - no explicit second-derivative kernel needed.")
print("- Limitations: approximation quality depends on the ratio between")
print("  the two sigma values (~1.6x is the standard choice).")
print("- Use cases: keypoint/blob detection at multiple scales, most famously")
print("  as the detector stage of the SIFT feature-matching algorithm.\n")

print("Canny Edge Detector")
print("- Strengths: the most complete pipeline - combines smoothing, gradient")
print("  computation, edge thinning (non-max suppression), and hysteresis")
print("  thresholding to produce clean, thin, well-connected edges with far")
print("  fewer false positives than the other methods.")
print("- Limitations: more parameters to tune (sigma, low/high thresholds),")
print("  more computationally expensive, can miss very low-contrast edges")
print("  if thresholds aren't tuned for the image.")
print("- Use cases: the standard choice for most practical edge-detection")
print("  tasks - object boundary detection, preprocessing for contour/shape")
print("  analysis, general computer vision pipelines.\n")

print("Overall summary:")
print("Sobel and Laplacian are simple, fast first- and second-derivative")
print("operators but are noise-sensitive on their own. LoG and DoG both add")
print("Gaussian smoothing to fix that (DoG being a fast approximation of")
print("LoG). Canny goes further, adding edge thinning and hysteresis on top")
print("of a Sobel gradient, which is why it consistently produces the")
print("cleanest, most usable edge maps of the five.")

plt.show()