from flask import Flask, render_template, request, send_file
from PIL import Image
import os
from rembg import remove

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

    # File conversion
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

    # Upscale image
    img = Image.open(file_path)
    new_size = (img.width * upscale_factor, img.height * upscale_factor)
    upscaled_img = img.resize(new_size, Image.Resampling.LANCZOS)

    output_path = os.path.join(UPLOAD_FOLDER, "upscaled.png")
    upscaled_img.save(output_path, "PNG")

    return send_file(output_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
