import os, re, shutil
from flask import Flask, render_template, request, jsonify
from PIL import Image
import pytesseract

app = Flask(__name__)
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "/usr/bin/tesseract"

def detect_fake_news(text):
    t = text.lower().strip()
    if len(t) < 10:
        return "REAL", 50, "Too short"

    # 1. SCAM / GOVT FREE GIFT
    if re.search(r'(free|muft).*(smartphone|laptop|recharge|money|gift)', t) and re.search(r'(aadhaar|bank|otp|whatsapp|claim)', t):
        return "FAKE", 99, "SCAM: Govt never asks Aadhaar/Bank on WhatsApp"

    # 2. SCIENCE IMPOSSIBLE - NASA / MOON / SUN / EARTH
    impossible_patterns = [
        (r'moon.*disappear', 'Moon cannot disappear for 24 hours - Impossible'),
        (r'sun.*disappear', 'Sun cannot disappear'),
        (r'earth.*stop.*rotat', 'Earth stopping rotation is impossible'),
        (r'nasa.*moon.*disappear', 'Fake NASA claim - Scientifically impossible'),
        (r'gravitational.*event.*stay indoors', 'No such gravitational event exists'),
        (r'nasa.*announced.*moon.*24 hours', 'Fake NASA announcement'),
        (r'scientists.*stay indoors.*moon', 'False scientific warning'),
    ]
    for pat, reason in impossible_patterns:
        if re.search(pat, t):
            return "FAKE", 98, f"Science Fake: {reason}"

    # 3. HEALTH MIRACLE FAKE
    if re.search(r'(drink|eat).*(cures| cure).*(cancer|diabetes|corona).*in.*(24 hours|1 day|7 days)', t):
        return "FAKE", 95, "Fake Health Claim: No single food cures disease in 24h"

    # 4. LOTTERY / MONEY
    if re.search(r'(you have won|lottery.*won|congratulations.*won).*\d+', t):
        return "FAKE", 97, "Lottery Scam"

    # 5. GENERAL CLICKBAIT + NO SOURCE
    fake_words = ['shocking', 'you wont believe', 'must share', 'viral', 'secret revealed']
    score = 0
    for w in fake_words:
        if w in t: score += 25
    
    # If it sounds like official news but has no official source and is impossible
    if ('nasa' in t or 'government of india' in t or 'who' in t) and ('.gov' not in t and 'official website' not in t):
        if len(t.split()) < 60: # short sensational news without source
            score += 20

    if score >= 25:
        return "FAKE", 90, "Sensational / Clickbait without official source"
    
    return "REAL", 85, "No fake pattern found - Looks authentic"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check():
    try:
        text_input = request.form.get('news_text','').strip()
        image_text = ""
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename != '':
                img = Image.open(file.stream).convert('L')
                image_text = pytesseract.image_to_string(img, lang='eng')
        final_text = text_input if text_input else image_text
        if not final_text.strip():
            return jsonify({'status':'error','message':'No text found'})
        label, conf, reason = detect_fake_news(final_text)
        return jsonify({'status':'success','label':label,'confidence':conf,'reason':reason,'extracted_text':final_text[:500]})
    except Exception as e:
        return jsonify({'status':'error','message':str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))