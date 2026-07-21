import cv2
import math

image = cv2.imread("../TASK1/image.jpg")

if image is None:
    print("Image not found!")
    exit()

# Get image dimensions
height, width, channels = image.shape

# asking for coordinate pairs
num_pairs = int(input("Enter the number of coordinate pairs: "))

for i in range(num_pairs):

    print(f"\nCoordinate Pair {i + 1}")
    while True:

        print("\nFirst Coordinate")

        x1 = int(input(f"Enter X1 (0 - {width - 1}): "))
        y1 = int(input(f"Enter Y1 (0 - {height - 1}): "))

        if 0 <= x1 < width and 0 <= y1 < height:
            break

        print("Invalid coordinates! Please try again.")

    while True:

        print("\nSecond Coordinate")

        x2 = int(input(f"Enter X2 (0 - {width - 1}): "))
        y2 = int(input(f"Enter Y2 (0 - {height - 1}): "))

        if 0 <= x2 < width and 0 <= y2 < height:
            break

        print("Invalid coordinates! Please try again.")

#formula
    dx = x2 - x1
    dy = y2 - y1

    manual_distance = (dx * dx + dy * dy) ** 0.5

  #verification
    verified_distance = math.sqrt(dx * dx + dy * dy)

   #dispalying results
    print("\nResult")
    print("----------------------------")
    print(f"First Coordinate  : ({x1}, {y1})")
    print(f"Second Coordinate : ({x2}, {y2})")
    print(f"Manual Distance   : {manual_distance:.2f}")
    print(f"Verified Distance : {verified_distance:.2f}")