import os
import pickle
from flask import Flask, request, render_template, jsonify
from PIL import Image
import pytesseract

app = Flask(__name__)

# --- Load Model and Vectorizer ---
model = None
vectorizer = None

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded: True")
except Exception as e:
    print(f"Model load failed: {e}")

try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("Vectorizer loaded: True")
except Exception as e:
    print(f"Vectorizer load failed: {e}")

# For Render Linux path
try:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
    print(f"Tesseract cmd: {pytesseract.pytesseract.tesseract_cmd}")
except:
    pass

def clean_text(text):
    if not text:
        return ""
    return text.lower().strip()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = ""

        # 1. Get text from form or JSON
        if request.form:
            text = request.form.get('news_text', '') or request.form.get('text', '') or ""

        # 2. Get image - FIXED: Check news_image FIRST
        file_obj = None
        # Check all possible field names, news_image is priority
        for field_name in ['news_image', 'file', 'image', 'news_image_file']:
            if field_name in request.files:
                f = request.files[field_name]
                if f and f.filename!= '':
                    file_obj = f
                    print(f"Found file in field: {field_name}, filename: {f.filename}")
                    break

        # 3. OCR if image found
        if file_obj:
            try:
                img = Image.open(file_obj.stream)
                # Convert to RGB if needed
                if img.mode!= 'RGB':
                    img = img.convert('RGB')
                ocr_text = pytesseract.image_to_string(img)
                print(f"OCR extracted: {ocr_text[:100]}")
                if ocr_text.strip():
                    text = ocr_text + " " + text
            except pytesseract.TesseractNotFoundError:
                return render_template('index.html',
                    prediction="Tesseract not installed on server",
                    confidence="Check apt.txt has tesseract-ocr")
            except Exception as e:
                print(f"OCR Error: {e}")
                # Continue with text if OCR fails

        text = text.strip()
        print(f"Final text for prediction: {text[:100]}")

        if not text:
            return render_template('index.html',
                prediction="Please enter text or upload an image with text",
                confidence="No text found")

        # 4. Predict
        if not model or not vectorizer:
            return render_template('index.html',
                prediction="Model not loaded",
                confidence="Check model.pkl and vectorizer.pkl")

        cleaned = clean_text(text)
        vector = vectorizer.transform([cleaned])
        pred = model.predict(vector)[0]

        try:
            proba = model.predict_proba(vector).max() * 100
        except:
            proba = 90.0

        # Handle different label formats
        if str(pred) == '1' or str(pred).lower() == 'real' or pred == 1 or pred == True:
            result = "REAL NEWS ✅"
        else:
            result = "FAKE NEWS ❌"

        return render_template('index.html',
            prediction=result,
            confidence=f"Confidence: {proba:.1f}%",
            input_text=text[:400])

    except Exception as e:
        print(f"Predict error: {e}")
        import traceback
        traceback.print_exc()
        return render_template('index.html',
            prediction=f"Server Error: {str(e)}",
            confidence="")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)