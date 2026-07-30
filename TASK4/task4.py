import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# Folder containing images
folder = "images"

# Read all image names
image_files = os.listdir(folder)

# Laplacian Kernel
kernel = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
])

# Threshold for Blur Detection
threshold = 100

# Store results
images = []
titles = []


# Process Each Image
for file in image_files:

    path = os.path.join(folder, file)

    # Read Color Image
    image = cv2.imread(path)

    if image is None:
        continue

    # Convert to RGB for display
    display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to Grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape

    # Create Laplacian Image
    laplacian = np.zeros((height, width), dtype=np.float32)

    # Manual Laplacian
    for i in range(1, height - 1):
        for j in range(1, width - 1):

            region = gray[i-1:i+2, j-1:j+2]

            value = np.sum(region * kernel)

            laplacian[i, j] = value


    # Variance of Laplacian
    score = np.var(laplacian)


    # Classification
    if score > threshold:
        result = "Sharp"
    else:
        result = "Blurry"

    print("--------------------------------")
    print("Image :", file)
    print("Variance :", round(score, 2))
    print("Classification :", result)

    images.append(display_image)

    titles.append(
        f"{file}\n"
        f"Score: {score:.2f}\n"
        f"{result}"
    )


# Display All Images Together
plt.figure(figsize=(20,5))

for i in range(len(images)):

    plt.subplot(1, len(images), i + 1)
    plt.imshow(images[i])
    plt.title(titles[i], fontsize=10)
    plt.axis("off")

plt.suptitle("Blur Detection using Variance of Laplacian", fontsize=16)

plt.tight_layout()

plt.show()


print("\n========== Blur Detection Completed ==========")
print("Images with higher Variance of Laplacian scores are classified as Sharp.")
print("Images with lower Variance of Laplacian scores are classified as Blurry.")