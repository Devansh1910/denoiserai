from flask import Flask, render_template, request
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 11, 6, 7, 21)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        image = cv2.imread(file_path)

        if image is None:
            return render_template('index.html', error='Unable to load the image')

        # Apply different types of noise
        image_with_gaussian_noise = add_gaussian_noise(image)
        image_with_white_noise = add_white_noise(image)
        image_with_brownian_noise = add_brownian_noise(image)
        image_with_impulse_noise = add_impulse_valued_noise(image)
        image_with_periodic_noise = add_periodic_noise(image)

        # Apply denoising
        denoised_image = denoise_image(image)

        # Save denoised image
        denoised_path = os.path.join(app.config['UPLOAD_FOLDER'], 'denoised_' + filename)
        cv2.imwrite(denoised_path, denoised_image)

        return render_template('result.html',
                               original_image=file_path,
                               gaussian_image=image_with_gaussian_noise,
                               white_image=image_with_white_noise,
                               brownian_image=image_with_brownian_noise,
                               impulse_image=image_with_impulse_noise,
                               periodic_image=image_with_periodic_noise,
                               denoised_image=denoised_path)

    else:
        return render_template('index.html', error='File type not allowed')

if __name__ == '__main__':
    app.run(debug=True)
