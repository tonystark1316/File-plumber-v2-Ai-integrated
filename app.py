import os
from flask import Flask, render_template, request, send_file
from PIL import Image
import torch
import cv2
import numpy as np
from realesrgan import RealESRGANer
from rembg import remove

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Disable gradients to save RAM
torch.set_grad_enabled(False)

# Load the ESRGAN general model with optimizations
def load_esrgan_model(model_path):
    return RealESRGANer(
        scale=2,  # Use 2x instead of 4x to reduce memory usage
        model_path=model_path,
        model=None,
        tile=512,  # Process images in 512x512 chunks to reduce RAM spikes
        tile_pad=10,
        pre_pad=0,
        half=True  # Use FP16 (half precision) to reduce RAM usage by 50%
    )

# Initialize the ESRGAN model
general_model = load_esrgan_model(os.path.join(MODEL_FOLDER, "realesrgan-x2plus.pth"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert():
    file = request.files['file']
    target_format = request.form['format']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Output path
    output_path = os.path.join(UPLOAD_FOLDER, f"converted.{target_format}")

    # Convert image format
    img = Image.open(file_path)
    img.save(output_path, format=target_format.upper())

    return send_file(output_path, as_attachment=True)

@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        # Background removal
        with open(file_path, "rb") as input_file:
            output = remove(input_file.read())

        output_path = os.path.join(UPLOAD_FOLDER, f"no_bg_{file.filename}")
        with open(output_path, "wb") as output_file:
            output_file.write(output)

        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return f"An error occurred: {e}"

@app.route('/upscale', methods=['POST'])
def upscale():
    file = request.files['file']
    upscale_factor = int(request.form['factor'])
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        # Load the image
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return "Failed to load the image."

        # Resize image before processing to prevent memory spikes
        MAX_SIZE = (1920, 1080)  # Limit input image size to 1080p
        img = cv2.resize(img, MAX_SIZE, interpolation=cv2.INTER_AREA)

        # Upscale the image using ESRGAN
        output, _ = general_model.enhance(img, outscale=upscale_factor)

        # Save the upscaled image
        output_path = os.path.join(UPLOAD_FOLDER, "upscaled.png")
        cv2.imwrite(output_path, output)

        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return f"An error occurred during upscaling: {e}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
