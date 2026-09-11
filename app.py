import os
import re
import shutil
from flask import Flask, render_template, request, jsonify
from PIL import Image
import pytesseract

app = Flask(__name__)

# Auto find tesseract path for Render Docker
tess_path = shutil.which("tesseract") or "/usr/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = tess_path

def detect_fake_news(text):
    text = text.strip()
    if len(text) < 10:
        return "REAL", 50, "Text too short"

    text_lower = text.lower()
    fake_keywords = ['shocking', 'you wont believe', '100% true', 'viral', 'breaking', 'urgent', 'must share', 'miracle', 'secret revealed', 'govt hiding']
    fake_score = 0
    reasons = []

    # Heuristic scoring
    if sum(1 for c in text if c.isupper()) / max(len(text),1) > 0.3:
        fake_score += 20
        reasons.append("Too many CAPS")
    if text.count('!') > 2 or text.count('?') > 3:
        fake_score += 20
        reasons.append("Too many!?")
    for kw in fake_keywords:
        if kw in text_lower:
            fake_score += 15

    # Numbers / clickbait check
    if re.search(r'\b\d{2,}\s*crore|\b\d{2,}\s*lakh', text_lower):
        if 'official' not in text_lower and 'source' not in text_lower:
            fake_score += 15

    if fake_score >= 40:
        confidence = min(95, 60 + fake_score)
        return "FAKE", confidence, ", ".join(reasons) if reasons else "Sensational language"
    else:
        confidence = min(95, 85 - fake_score)
        return "REAL", confidence, "Looks authentic"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check():
    try:
        text_input = request.form.get('news_text', '').strip()
        image_text = ""

        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename!= '':
                img = Image.open(file.stream)
                # Improve OCR
                img = img.convert('L') # grayscale
                image_text = pytesseract.image_to_string(img, lang='eng')

        final_text = text_input if text_input else image_text

        if not final_text.strip():
            return jsonify({'status': 'error', 'message': 'Could not read text from image. Try clearer image or type text.'})

        label, conf, reason = detect_fake_news(final_text)

        return jsonify({
            'status': 'success',
            'label': label,
            'confidence': conf,
            'reason': reason,
            'extracted_text': final_text[:500]
        })

    except Exception as e:
        # Handle tesseract not installed error gracefully
        err = str(e).lower()
        if 'tesseract' in err:
            return jsonify({'status': 'error', 'message': f"OCR Error: {str(e)}. Try clearer image or type text."})
        return jsonify({'status': 'error', 'message': f"Error: {str(e)}"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
