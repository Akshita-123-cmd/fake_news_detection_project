from flask import Flask, render_template, request
import pickle
from PIL import Image
import pytesseract
import os
import platform
import shutil

app = Flask(__name__)

# ========== FIX FOR BOTH LAPTOP AND RENDER ==========
if platform.system() == "Windows":
    # LAPTOP PATH
    win_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(win_path):
        pytesseract.pytesseract.tesseract_cmd = win_path
        print(f"✅ Running on WINDOWS LAPTOP - {win_path}")
    else:
        # try alternate location
        alt_path = r'C:\Users\diddi\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
        if os.path.exists(alt_path):
            pytesseract.pytesseract.tesseract_cmd = alt_path
            print(f"✅ Running on WINDOWS LAPTOP - {alt_path}")
        else:
            print("❌ Tesseract NOT FOUND on laptop! Reinstall it!")
else:
    # RENDER - LINUX - Don't set path, auto-find
    print("✅ Running on RENDER LINUX")

print(f"Tesseract in PATH? : {shutil.which('tesseract')}")
print(f"Tesseract cmd used: {pytesseract.pytesseract.tesseract_cmd}")

# Load model
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
                img = Image.open(file.stream).convert('RGB')
                if max(img.size) > 2000:
                    img.thumbnail((2000, 2000))
                text_from_image = pytesseract.image_to_string(img)
                print(f"OCR SUCCESS: {text_from_image[:200]}")
            except Exception as e:
                print(f"OCR FAIL: {e}")
                return render_template('index.html',
                    prediction=f"❌ Image read failed: {e}",
                    color="red",
                    news=manual_text,
                    text_extracted=str(e))

    final_text = text_from_image if len(text_from_image.strip()) > 3 else manual_text

    if not final_text.strip():
        return render_template('index.html',
            prediction="❌ Upload clearer image or type text!",
            color="red",
            news=manual_text,
            text_extracted=f"OCR empty. Extracted: '{text_from_image}'")

    vec = vectorizer.transform([final_text])
    pred = model.predict(vec)[0]
    result = "✅ REAL NEWS" if pred == 1 else "❌ FAKE NEWS"
    color = "green" if pred == 1 else "red"
    return render_template('index.html', prediction=result, color=color, news=manual_text, text_extracted=final_text[:500])

if __name__ == '__main__':
    app.run(debug=True)