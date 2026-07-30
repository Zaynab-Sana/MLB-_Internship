import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# Folder containing images
folder = "images"

# Read image names
image_files = sorted(os.listdir(folder))

# Laplacian Kernel
kernel = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
])

# Blur Threshold
threshold = 100

# Lists for display
images = []
titles = []

# Comparison Table
results = []


# Process Images
for file in image_files:

    path = os.path.join(folder, file)

    image = cv2.imread(path)

    if image is None:
        continue

    display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape

    laplacian = np.zeros((height, width), dtype=np.float32)

    # Manual Laplacian
    for i in range(1, height - 1):
        for j in range(1, width - 1):

            region = gray[i-1:i+2, j-1:j+2]

            value = np.sum(region * kernel)

            laplacian[i, j] = value

    # Variance
    score = np.var(laplacian)

    # Classification
    if score > threshold:
        level = "Sharp"
        observation = "Clear edges and high details"
    else:
        level = "Blurry"
        observation = "Weak edges and smooth appearance"

    # Store Results
    results.append([file, round(score,2), level, observation])

    images.append(display)

    titles.append(f"{file}\n{level}")


# Display Images
plt.figure(figsize=(18,8))

for i in range(len(images)):

    plt.subplot(2,5,i+1)

    plt.imshow(images[i])

    plt.title(titles[i],fontsize=9)

    plt.axis("off")

plt.suptitle("Blur Metrics Analysis",fontsize=16)

plt.tight_layout()

# Comparison Table
print("\n================ Blur Metrics Comparison Table ================\n")

print("{:<15} {:<20} {:<12} {}".format(
    "Image Name",
    "Variance Score",
    "Blur Level",
    "Observation"
))

print("-"*80)

for row in results:

    print("{:<15} {:<20} {:<12} {}".format(
        row[0],
        row[1],
        row[2],
        row[3]
    ))


# Analysis
print("\n================ Analysis ================\n")

print("1. Images with higher Variance of Laplacian scores contain stronger edges and are classified as Sharp.")
print("2. Images with lower scores contain fewer edges and are classified as Blurry.")
print("3. As blur increases, the Variance of Laplacian score decreases.")
print("4. Variance of Laplacian is an effective metric for measuring image sharpness.")

plt.show()


