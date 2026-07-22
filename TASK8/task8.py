import cv2
import numpy as np
import matplotlib.pyplot as plt



image = cv2.imread("../TASK6/image3.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found!")
    exit()


# defining kernel
kernel = [
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
]


#covulation with stride
def convolution_with_stride(image, kernel, stride):

    height, width = image.shape

    kernel_size = 3

    # Calculate output size
    output_height = ((height - kernel_size) // stride) + 1
    output_width = ((width - kernel_size) // stride) + 1


    #creating an output image
    output = np.zeros((output_height, output_width), dtype=np.uint8)


    output_row = 0

    for row in range(0, height - kernel_size + 1, stride):

        output_col = 0

        for col in range(0, width - kernel_size + 1, stride):

            total = 0


            # Visit kernel pixels
            for i in range(3):
                for j in range(3):

                    pixel = int(image[row+i][col+j])

                    kernel_value = kernel[i][j]

                    total += pixel * kernel_value


            # Limit value
            if total < 0:
                total = 0

            if total > 255:
                total = 255


            output[output_row][output_col] = int(total)

            output_col += 1


        output_row += 1


    return output



# Apply stride 1
output_stride1 = convolution_with_stride(image, kernel, 1)
# Apply stride 2
output_stride2 = convolution_with_stride(image, kernel, 2)

# Display results

plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
plt.imshow(image,cmap="gray")
plt.title("Original Image")
plt.axis("off")


plt.subplot(1,3,2)
plt.imshow(output_stride1,cmap="gray")
plt.title(f"Stride 1\nSize: {output_stride1.shape}")
plt.axis("off")


plt.subplot(1,3,3)
plt.imshow(output_stride2,cmap="gray")
plt.title(f"Stride 2\nSize: {output_stride2.shape}")
plt.axis("off")


plt.show()



print("Original Image Size:", image.shape)
print("Stride 1 Output Size:", output_stride1.shape)
print("Stride 2 Output Size:", output_stride2.shape)


print("\nExplanation:")
print("-----------------------------")
print("Stride 1 moves the kernel one pixel at a time.")
print("It produces a larger output and captures more details.")

print("\nStride 2 moves the kernel two pixels at a time.")
print("It reduces the output size and skips some positions.")
print("It is faster but may lose some image information.")