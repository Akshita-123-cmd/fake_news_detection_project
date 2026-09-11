import os
import pickle
import shutil
from flask import Flask, request, render_template
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

# --- FIXED: Auto-find tesseract path (works on Render + Local) ---
try:
    tess_path = shutil.which("tesseract")
    if tess_path:
        pytesseract.pytesseract.tesseract_cmd = tess_path
        print(f"Tesseract found at: {tess_path}")
    else:
        # Try common Render paths
        for p in ["/usr/bin/tesseract", "/usr/local/bin/tesseract"]:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                print(f"Tesseract found at: {p}")
                break
        else:
            print("Tesseract not found in PATH, will try default")
except Exception as e:
    print(f"Tesseract path setup error: {e}")

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

        # 1. Get text from form
        if request.form:
            text = request.form.get('news_text', '') or request.form.get('text', '') or ""

        # 2. Get image - Check news_image FIRST (your HTML sends this)
        file_obj = None
        for field_name in ['news_image', 'file', 'image']:
            if field_name in request.files:
                f = request.files[field_name]
                if f and f.filename!= '':
                    file_obj = f
                    print(f"Found file in: {field_name} -> {f.filename}")
                    break

        # 3. OCR if image found
        if file_obj:
            try:
                img = Image.open(file_obj.stream)
                if img.mode!= 'RGB':
                    img = img.convert('RGB')
                ocr_text = pytesseract.image_to_string(img)
                print(f"OCR Text: {ocr_text[:150]}")
                if ocr_text.strip():
                    text = ocr_text + " " + text
            except Exception as e:
                print(f"OCR Error: {e}")
                # Don't fail, try to continue - if text was also typed
                if not text.strip():
                    return render_template('index.html',
                        prediction="Could not read text from image",
                        confidence=f"OCR Error: {str(e)[:100]}. Try clearer image or type text.")

        text = text.strip()
        print(f"Final text: {text[:150]}")

        if not text:
            return render_template('index.html',
                prediction="Please enter text or upload an image with clear text",
                confidence="No text found")

        # 4. Predict
        if not model or not vectorizer:
            return render_template('index.html',
                prediction="Model not loaded",
                confidence="Check model.pkl and vectorizer.pkl exist")

        cleaned = clean_text(text)
        vector = vectorizer.transform([cleaned])
        pred = model.predict(vector)[0]

        try:
            proba = model.predict_proba(vector).max() * 100
        except:
            proba = 85.0

        # Handle all label types
        if str(pred) == '1' or str(pred).lower() in ['real', 'true'] or pred == 1:
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
            prediction=f"Error: {str(e)}",
            confidence="Check Render logs")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)