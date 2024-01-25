from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
from io import BytesIO
from base64 import b64encode

app = Flask(__name__)

def add_gaussian_noise(image, mean=0, sigma=25):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_white_noise(image, strength=25):
    row, col, ch = image.shape
    noise = np.random.randint(-strength, strength, (row, col, ch))
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_brownian_noise(image, strength=25):
    row, col, ch = image.shape
    brown = np.cumsum(np.cumsum(np.random.randn(row, col, ch), axis=0), axis=1)
    noisy = image + strength * brown
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_impulse_valued_noise(image, probability=0.01, strength=255):
    row, col, ch = image.shape
    noisy = image.copy()
    mask = np.random.rand(row, col, ch) < probability
    noisy[mask] = np.random.randint(0, 2, mask.sum()) * strength
    return noisy.astype(np.uint8)

def add_periodic_noise(image, frequency=0.1, strength=25):
    row, col, ch = image.shape
    x = np.arange(0, row)
    y = np.arange(0, col)
    X, Y = np.meshgrid(x, y)
    noise = strength * np.sin(2 * np.pi * frequency * X / row + 2 * np.pi * frequency * Y / col)
    noisy = image + noise.T[:, :, np.newaxis]
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_gamma_noise(image, alpha=1, beta=1):
    row, col, ch = image.shape
    gamma_noise = np.random.gamma(alpha, beta, (row, col, ch))
    noisy = image * gamma_noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_structured_noise(image, strength=25):
    row, col, ch = image.shape
    structured_noise = strength * np.sin(np.linspace(0, 2 * np.pi, col))
    structured_noise = structured_noise.reshape(1, col, 1)
    noisy = image + structured_noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def process_image(image, processing_option):
    if processing_option == 'original':
        return image
    elif processing_option == 'gaussian_noise':
        return add_gaussian_noise(image)
    elif processing_option == 'white_noise':
        return add_white_noise(image)
    elif processing_option == 'brownian_noise':
        return add_brownian_noise(image)
    elif processing_option == 'impulse_valued_noise':
        return add_impulse_valued_noise(image)
    elif processing_option == 'periodic_noise':
        return add_periodic_noise(image)
    elif processing_option == 'gamma_noise':
        return add_gamma_noise(image)
    elif processing_option == 'structured_noise':
        return add_structured_noise(image)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image_route():
    if 'image' not in request.files:
        return render_template('error.html', message="No image file provided.")

    image_file = request.files['image']

    try:
        # Process the image
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Get the selected processing option
        processing_option = request.form.get('processing_option', 'original')

        # Process the image based on the selected option
        processed_image = process_image(image, processing_option)

        # Encode the processed image as base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        result_image_base64 = b64encode(buffer).decode('utf-8')

        # Render the result template with the processed image
        return render_template('result.html', result_image=result_image_base64)

    except Exception as e:
        return render_template('error.html', message=f"Error processing image: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)