from flask import Flask, render_template, request, send_file
from PIL import Image
import os
from rembg import remove
import cv2
import torch
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet  # Required for loading ESRGAN model

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
MODEL_PATH = os.path.abspath(os.path.join(MODEL_FOLDER, "realesrgan-x4plus.pth"))

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the ESRGAN model correctly
def load_esrgan_model(model_path):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    # Load the model file and fix any key mismatches
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Some pre-trained models have 'params_ema', check if we need to extract it
    if "params_ema" in state_dict:
        state_dict = state_dict["params_ema"]
    
    model.load_state_dict(state_dict, strict=False)  # strict=False to avoid missing keys issue

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False
    )
    return upsampler

# Initialize the general ESRGAN model
general_model = load_esrgan_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/templates/about.html')
def about():
    return render_template('about.html')

@app.route('/templates/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/convert', methods=['POST'])
def convert():
    file = request.files['file']
    target_format = request.form['format']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    output_path = os.path.join(UPLOAD_FOLDER, f"converted.{target_format}")
    if target_format in ['png', 'jpg', 'jpeg', 'bmp', 'webp', 'ico', 'gif']:
        img = Image.open(file_path)
        img.save(output_path, format=target_format.upper())
    elif target_format == 'pdf':
        img = Image.open(file_path).convert("RGB")
        img.save(output_path, "PDF")
    else:
        return "Unsupported format. SVG conversion is not implemented."
    return send_file(output_path, as_attachment=True)

@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
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
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return "Failed to load the image."

        output, _ = general_model.enhance(img, outscale=upscale_factor)

        output_path = os.path.join(UPLOAD_FOLDER, "upscaled.png")
        cv2.imwrite(output_path, output)

        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return f"An error occurred during upscaling: {e}"

if __name__ == '__main__':
    app.run(debug=True)
