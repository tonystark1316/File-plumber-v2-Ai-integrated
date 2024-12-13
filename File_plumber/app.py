from flask import Flask, render_template, request, send_file
from PIL import Image
import os
from rembg import remove
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


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

    # Open the file and convert
    output_path = os.path.join(UPLOAD_FOLDER, f"converted.{target_format}")
    if target_format in ['png', 'jpg', 'jpeg', 'bmp', 'webp', 'ico', 'gif']:
        img = Image.open(file_path)
        img.save(output_path, format=target_format.upper())
    elif target_format == 'pdf':
        img = Image.open(file_path).convert("RGB")
        img.save(output_path, "PDF")
    elif target_format == 'svg':
        return "SVG conversion is not yet implemented."
    return send_file(output_path, as_attachment=True)


@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Remove background
    with open(file_path, "rb") as input_file:
        output = remove(input_file.read())

    output_path = os.path.join(UPLOAD_FOLDER, "no_bg.png")
    with open(output_path, "wb") as output_file:
        output_file.write(output)

    return send_file(output_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
