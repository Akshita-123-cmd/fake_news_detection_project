from flask import Flask, render_template, request
import pickle
from PIL import Image
import pytesseract
import os

app = Flask(__name__)
import platform
import shutil

# SMART FIX - Only set path on Windows laptop
# On Render (Linux), DON'T set any path, let it auto-find
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    print("Running on Windows")
else:
    print("Running on Render Linux - using system tesseract")

print(f"Tesseract found in PATH: {shutil.which('tesseract')}")
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
print("Model loaded!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text_from_image = ""
    manual_text = request.form.get('news','')

    if 'news_image' in request.files:
        file = request.files['news_image']
        if file and file.filename!= '':
            try:
                img = Image.open(file.stream)
                # Fix for mobile large images
                img = img.convert('RGB')
                # Resize if too big (mobile camera images are huge)
                max_size = 2000
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size))
                text_from_image = pytesseract.image_to_string(img)
                print(f"MOBILE OCR SUCCESS: {text_from_image[:100]}")
            except Exception as e:
                print(f"MOBILE OCR FAIL: {e}")
                return render_template('index.html', prediction=f"❌ Image read failed on server: {e}", color="red", news=manual_text, text_extracted=str(e))

    final_text = text_from_image if len(text_from_image.strip()) > 3 else manual_text

    if not final_text.strip():
        return render_template('index.html', prediction="❌ Please upload clearer image with bigger text!", color="red", news=manual_text, text_extracted=f"OCR empty. Got: '{text_from_image}'")

    vec = vectorizer.transform([final_text])
    pred = model.predict(vec)[0]
    result = "✅ REAL NEWS" if pred == 1 else "❌ FAKE NEWS"
    color = "green" if pred == 1 else "red"
    return render_template('index.html', prediction=result, color=color, news=manual_text, text_extracted=final_text[:400])

if __name__ == '__main__':
    app.run(debug=True)