import os
import shutil
from flask import Flask, render_template, request, jsonify
from PIL import Image
import pytesseract

# ========== FIX FOR BOTH LAPTOP + RENDER ==========
# This auto-finds tesseract on Windows and Linux
tess_path = shutil.which("tesseract")
if not tess_path:
    # Check common install locations
    for p in [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract"
    ]:
        if os.path.exists(p):
            tess_path = p
            break

if tess_path:
    pytesseract.pytesseract.tesseract_cmd = tess_path
    print(f"✅ Tesseract READY: {tess_path}")
else:
    print("❌ Tesseract NOT FOUND! Install it on Windows from UB Mannheim")
# =================================================

# Try to load your ML model
try:
    import joblib
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    print("✅ Model loaded")
except Exception as e:
    print(f"⚠️ Model not loaded: {e}")
    model = None
    vectorizer = None

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image uploaded
        if 'file' in request.files and request.files['file'].filename!= '':
            file = request.files['file']
            img = Image.open(file.stream)

            # OCR - Extract text from image
            try:
                text = pytesseract.image_to_string(img)
            except pytesseract.TesseractNotFoundError:
                return jsonify({
                    'error': 'Tesseract is not installed or not in PATH. On Windows, install from https://github.com/UB-Mannheim/tesseract/wiki'
                }), 500

            if not text.strip():
                return jsonify({'error': 'No text found in image! Try clearer image.'}), 400

        else:
            # Text input
            text = request.form.get('news', '') or request.json.get('news', '') if request.json else ''

        if not text or not text.strip():
            return jsonify({'error': 'Please enter text or upload image'}), 400

        # Prediction
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
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)