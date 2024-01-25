import cv2
import numpy as np
import matplotlib.pyplot as plt

# Take image path as input from the user
image_path = input("Enter the path to the image: ")

# Load the image
image = cv2.imread(image_path)

# Check if the image is loaded successfully
if image is None:
    print(f"Error: Unable to load the image from {image_path}")
    exit()

# Check image type and convert if necessary
if image.dtype != np.uint8:
    image = image.astype(np.uint8)

# Check the number of channels
if len(image.shape) == 2:
    # Convert to 3 channels if it's a grayscale image
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Apply fastNlMeansDenoisingColored
dst = cv2.fastNlMeansDenoisingColored(image, None, 11, 6, 7, 21)

# Set up the plot
row, col = 1, 2
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.suptitle('Image Denoising Comparison', fontsize=16)

# Plot the original image
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')

# Plot the denoised image
axs[1].imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
axs[1].set_title('Fast Means Denoising')

# Adjust layout
plt.subplots_adjust(top=0.85, wspace=0.3)

# Show the plot
plt.show()
