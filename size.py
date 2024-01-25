import cv2
import os

def denoise_image(image_path, save_path, input_size=(800, 600), output_size=(400, 300)):
    # Read the image and resize it
    noisy_image = cv2.imread(image_path)
    noisy_image = cv2.resize(noisy_image, input_size)

    # Check if the image is loaded successfully
    if noisy_image is None:
        print(f"Error: Unable to load the image from {image_path}")
        return None

    # Get the size of the original image
    original_size = os.path.getsize(image_path)

    # Apply GaussianBlur for denoising
    denoised_image = cv2.GaussianBlur(noisy_image, (5, 5), 0)

    # Resize the denoised image
    denoised_image = cv2.resize(denoised_image, output_size)

    # Save the denoised image
    cv2.imwrite(save_path, denoised_image)

    # Get the size of the denoised image
    denoised_size = os.path.getsize(save_path)

    return denoised_image, original_size, denoised_size

# Example usage
input_image_path = 'uploads/input.jpg'
output_image_path = 'D:/Subjects/Semester 5/BT-3588/Image/Test/denoised_image.jpg'

# Set the desired input and output sizes (width, height)
input_size = (400, 300)
output_size = (400, 350)

denoised_img, original_size, denoised_size = denoise_image(input_image_path, output_image_path, input_size, output_size)

# Display the original and denoised images if denoising was successful
if denoised_img is not None:
    cv2.imshow('Noisy Image', cv2.resize(cv2.imread(input_image_path), input_size))
    cv2.imshow('Denoised Image', denoised_img)

    # Display the size of the original and denoised images
    print(f"Original Image Size: {original_size} bytes")
    print(f"Denoised Image Size: {denoised_size} bytes")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
