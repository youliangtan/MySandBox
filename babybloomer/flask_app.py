from flask import Flask, request, render_template, send_from_directory
import os
from mp_model import HazardDetector
import json
import cv2

app = Flask(__name__)

# Ensure there's a folder to save the uploaded files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = HazardDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("called upload_file")
    if 'image' not in request.files:
        return 'No image part'
    file = request.files['image']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        img, results = model.detect(file_path)
        json_str = json.dumps(results)
        name, ext = os.path.splitext(filename)
        edited_filename = f"{name}-edit{ext}"
        
        # resize image to 640x480
        img = cv2.resize(img, (640, 480))

        # save image in the uploads folder with cv2
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], edited_filename), img)
        print("saved image")
        suggestions = "Suggestions: \n - amazon link: https://tinyurl.com/3j9z9w3u \n - Wrap it up."
        return 'Image uploaded successfully: \n\n Harzards\n' + json_str + '\n\n' + suggestions

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    print("  called uploaded_file")
    print(filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/templates/<filename>')
def template_assets(filename):
    print("  called template")
    print(filename)
    return send_from_directory("templates", filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
