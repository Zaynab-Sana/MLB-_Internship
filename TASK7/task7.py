#===============TASK 7 =======================

import cv2

# Load image
image = cv2.imread("../TASK2/modifiedimage2.png")

# Validation
if image is None:
    print("Image Not Found")
    exit()

# Get image dimensions
height = image.shape[0]
width = image.shape[1]
channels = image.shape[2]

# Get bit depth
data_type = image.dtype

# Calculate bits per channel
bits_per_channel = image.itemsize * 8

# Convert bits to bytes
bytes_per_channel = bits_per_channel / 8

# Calculate memory in bytes
memory_bytes = height * width * channels * bytes_per_channel

# Convert bytes to MB
memory_MB = memory_bytes / (1024 * 1024)

# Display information
print("Image Height:", height)
print("Image Width:", width)
print("Number of Channels:", channels)
print("Data Type:", data_type)
print("Bit Depth:", bits_per_channel, "bits")
print("Memory Occupied:", memory_bytes, "bytes")
print(f"Memory Occupied: {memory_MB:.2f} MB")