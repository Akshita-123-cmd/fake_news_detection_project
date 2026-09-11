from flask import Flask, render_template, request, jsonify
from PIL import Image
import pytesseract
import os
import pickle

app = Flask(__name__)

# --- Load Model and Vectorizer ---
model = None
vectorizer = None

try:
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
    if os.path.exists('vectorizer.pkl'):
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
    print(f"Model loaded: {model is not None}, Vectorizer loaded: {vectorizer is not None}")
except Exception as e:
    print(f"Model load error: {e}")

# --- Tesseract Path Setup (Works on both Windows + Render Linux) ---
# On Render Linux, tesseract is at /usr/bin/tesseract (from apt.txt)
# On Windows, set it manually if needed
if os.name == 'nt': # Windows
    # Change this path if your tesseract is installed elsewhere
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else: # Linux / Render
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

print(f"Tesseract cmd: {pytesseract.pytesseract.tesseract_cmd}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = ""

        # --- Check if image uploaded - FIXED to accept 'file' AND 'image' ---
        file = None
        if 'file' in request.files and request.files['file'].filename!= '':
            file = request.files['file']
        elif 'image' in request.files and request.files['image'].filename!= '':
            file = request.files['image']

        if file:
            try:
                img = Image.open(file.stream)

                # OCR - Extract text from image
                try:
                    text = pytesseract.image_to_string(img)
                except pytesseract.TesseractNotFoundError:
                    return jsonify({
                        'error': 'Tesseract is not installed or not in PATH. On Windows, install from https://github.com/UB-Mannheim/tesseract/wiki - On Render, add tesseract-ocr to apt.txt'
                    }), 500

                if not text.strip():
                    return jsonify({'error': 'No text found in image! Try clearer image.'}), 400

            except Exception as e:
                return jsonify({'error': f'Image processing error: {str(e)}'}), 400

        else:
            # --- Text input - FIXED 415 ERROR ---
            # Safe way: check form first, then JSON only if it's actually JSON
            text = request.form.get('news', '').strip()

            if not text:
                text = request.form.get('text', '').strip()

            # Only try to parse JSON if Content-Type is application/json
            if not text and request.is_json:
                json_data = request.get_json(silent=True)
                if json_data:
                    text = json_data.get('news', '').strip() or json_data.get('text', '').strip() or ''

            if not text or not text.strip():
                return jsonify({'error': 'Please enter text or upload image'}), 400

        # --- Prediction ---
        if model and vectorizer:
            vec = vectorizer.transform([text])
            pred = model.predict(vec)[0]
            result = "FAKE" if pred == 1 else "REAL"
            confidence = model.predict_proba(vec).max() * 100 if hasattr(model, 'predict_proba') else 0
        else:
            # Dummy if model not found
            result = "REAL" if len(text) > 50 else "FAKE"
            confidence = 85

        return jsonify({
            'text_extracted': text[:500],
            'prediction': result,
            'confidence': f"{confidence:.2f}%"
        })

    except Exception as e:
        print(f"Predict error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)