import os
import re
import shutil
from flask import Flask, render_template, request, jsonify
from PIL import Image
import pytesseract

app = Flask(__name__)

tess_path = shutil.which("tesseract") or "/usr/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = tess_path

def detect_fake_news(text):
    text = text.strip()
    if len(text) < 10:
        return "REAL", 50, "Text too short"

    t = text.lower()

    # --- HIGH RISK SCAM PATTERNS (100% FAKE) ---
    scam_rules = [
        (r'free.*(smartphone|laptop|recharge|money|gift|prize)', 'Free gift claim'),
        (r'(aadhaar|aadhar).*(bank|account|otp|whatsapp)', 'Asking Aadhaar + Bank'),
        (r'(bank details|bank account|otp).*(whatsapp|send)', 'Asking Bank on WhatsApp'),
        (r'government.*free.*(phone|laptop|money|yojana).*whatsapp', 'Govt free offer on WhatsApp'),
        (r'send.*(aadhaar|bank).*(whatsapp|claim)', 'Send personal data on WhatsApp'),
        (r'click.*link.*claim', 'Click link to claim'),
        (r'lottery.*won|you have won.*\d+', 'Lottery scam'),
    ]

    for pattern, reason in scam_rules:
        if re.search(pattern, t):
            return "FAKE", 98, f"SCAM Alert: {reason} - Govt never asks bank details on WhatsApp"

    # --- GENERAL FAKE KEYWORDS ---
    fake_keywords = ['shocking', 'you wont believe', 'viral', 'must share', 'secret revealed', '100% true', 'forward to', 'share with 10']
    score = 0
    reasons = []
    
    for kw in fake_keywords:
        if kw in t:
            score += 20
            reasons.append(kw)

    if 'free' in t and ('government' in t or 'govt' in t or 'modi' in t):
        if 'official website' not in t and '.gov.in' not in t:
            score += 40
            reasons.append("Free govt offer without official link")

    if score >= 30:
        return "FAKE", min(95, 70+score), ", ".join(reasons)

    return "REAL", 85, "Looks authentic - No scam pattern found"

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
                img = Image.open(file.stream).convert('L')
                image_text = pytesseract.image_to_string(img, lang='eng')
        final_text = text_input if text_input else image_text
        if not final_text.strip():
            return jsonify({'status': 'error', 'message': 'Could not read text'})
        label, conf, reason = detect_fake_news(final_text)
        return jsonify({'status': 'success','label': label,'confidence': conf,'reason': reason,'extracted_text': final_text[:500]})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)