import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, sigma=25):
    # Add Gaussian noise to the image
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_white_noise(image, strength=25):
    # Add white noise to the image
    row, col, ch = image.shape
    noise = np.random.randint(-strength, strength, (row, col, ch))
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_brownian_noise(image, strength=25):
    # Add Brownian noise to the image
    row, col, ch = image.shape
    brown = np.cumsum(np.cumsum(np.random.randn(row, col, ch), axis=0), axis=1)
    noisy = image + strength * brown
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_impulse_valued_noise(image, probability=0.01, strength=255):
    # Add impulse-valued noise to the image
    row, col, ch = image.shape
    noisy = image.copy()
    mask = np.random.rand(row, col, ch) < probability
    noisy[mask] = np.random.randint(0, 2, mask.sum()) * strength
    return noisy.astype(np.uint8)

def add_periodic_noise(image, frequency=0.1, strength=25):
    # Add periodic noise to the image
    row, col, ch = image.shape
    x = np.arange(0, row)
    y = np.arange(0, col)
    X, Y = np.meshgrid(x, y)
    noise = strength * np.sin(2 * np.pi * frequency * X / row + 2 * np.pi * frequency * Y / col)
    noisy = image + noise.T[:, :, np.newaxis]  # Transpose noise array
    return np.clip(noisy, 0, 255).astype(np.uint8)

# Load the image
image_path = r'Denoising/templates/Picture/300px-Kodim17_noisy.jpg'
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

# Apply different types of noise
image_with_gaussian_noise = add_gaussian_noise(image)
image_with_white_noise = add_white_noise(image)
image_with_brownian_noise = add_brownian_noise(image)
image_with_impulse_noise = add_impulse_valued_noise(image)
image_with_periodic_noise = add_periodic_noise(image)

# Apply fastNlMeansDenoisingColored
dst = cv2.fastNlMeansDenoisingColored(image, None, 11, 6, 7, 21)

# Set up the plot
row, col = 3, 3  # Adjust as needed
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.suptitle('DenoiserAI', fontsize=16)

# Plot the original image
axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Original Image')

# Plot the images with different types of noise
axs[0, 1].imshow(cv2.cvtColor(image_with_gaussian_noise, cv2.COLOR_BGR2RGB))
axs[0, 1].set_title('Gaussian Noise')

axs[0, 2].imshow(cv2.cvtColor(image_with_white_noise, cv2.COLOR_BGR2RGB))
axs[0, 2].set_title('White Noise')

axs[1, 0].imshow(cv2.cvtColor(image_with_brownian_noise, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title('Brownian Noise')

axs[1, 1].imshow(cv2.cvtColor(image_with_impulse_noise, cv2.COLOR_BGR2RGB))
axs[1, 1].set_title('Impulse-valued Noise')

axs[1, 2].imshow(cv2.cvtColor(image_with_periodic_noise, cv2.COLOR_BGR2RGB))
axs[1, 2].set_title('Periodic Noise')

# Plot the denoised image
axs[2, 0].imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
axs[2, 0].set_title('Denoised Image')

# Adjust layout
plt.subplots_adjust(top=0.85, wspace=0.3, hspace=0.5)

# Show the plot
plt.show()
