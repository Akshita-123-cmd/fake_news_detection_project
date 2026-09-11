from flask import Flask, render_template, request
import pickle
from PIL import Image
import pytesseract
import os

app = Flask(__name__)

# Load model
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    print("Model loaded!")
except:
    model = None
    vectorizer = None
    print("Model not found - using dummy")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = ""

    # 1. Check if image uploaded
    if 'news_image' in request.files:
        file = request.files['news_image']
        if file and file.filename!= '':
            try:
                img = Image.open(file.stream)
                # OCR - extract text from image
                text = pytesseract.image_to_string(img)
                print(f"OCR Text: {text}")
            except Exception as e:
                print(f"OCR Error: {e}")
                text = ""

    # 2. If no image text, take textarea text
    if not text or len(text.strip()) < 5:
        text = request.form.get('news', '')

    if not text or len(text.strip()) < 5:
        return render_template('index.html', prediction="Please enter text or upload a clear image!", news=text, text_extracted="")

    # 3. Predict
    try:
        if model and vectorizer:
            vec = vectorizer.transform([text])
            pred = model.predict(vec)[0]
            # model: 0 = Fake, 1 = Real (change if opposite)
            if pred == 1:
                result = "✅ REAL NEWS"
                color = "green"
            else:
                result = "❌ FAKE NEWS"
                color = "red"
        else:
            # fallback
            result = "✅ REAL NEWS (Demo Mode - Model not loaded)"
            color = "green"

        return render_template('index.html', prediction=result, color=color, news=request.form.get('news',''), text_extracted=text[:300])

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {e}", news=text)

if __name__ == '__main__':
    app.run(debug=True)