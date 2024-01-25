import cv2
import numpy as np
import matplotlib.pyplot as plt
import firebase_admin
from firebase_admin import credentials, storage

# Initialize Firebase (replace 'path/to/serviceAccountKey.json' with your actual file)
cred = credentials.Certificate('path/to/serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'your-storage-bucket-url.appspot.com'
})

# Function to upload image to Firebase Storage
def upload_to_firebase(image):
    bucket = storage.bucket()
    blob = bucket.blob("images/image.jpg")  # Replace 'image.jpg' with your desired filename
    _, img_encoded = cv2.imencode('.jpg', image)
    blob.upload_from_string(img_encoded.tobytes(), content_type='image/jpg')
    print("Image uploaded to Firebase Storage")

# Function to download image from Firebase Storage
def download_from_firebase():
    bucket = storage.bucket()
    blob = bucket.blob("images/image.jpg")  # Replace 'image.jpg' with the same filename used for upload
    image_bytes = blob.download_as_bytes()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_np

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

# Upload the original image to Firebase Storage
upload_to_firebase(image)

# Download the image from Firebase Storage
image_from_firebase = download_from_firebase()

# Perform denoising on the downloaded image
dst_from_firebase = cv2.fastNlMeansDenoisingColored(image_from_firebase, None, 11, 6, 7, 21)

# Plot the denoised image from Firebase
plt.imshow(cv2.cvtColor(dst_from_firebase, cv2.COLOR_BGR2RGB))
plt.title('Denoised Image from Firebase')
plt.show()
