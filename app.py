import os, re, shutil
from flask import Flask, render_template, request, jsonify
from PIL import Image
import pytesseract

app = Flask(__name__)
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "/usr/bin/tesseract"

def detect_fake_news(text):
    t = text.lower()
    if "free" in t and "smartphone" in t and "whatsapp" in t and ("aadhaar" in t or "bank" in t):
        return "FAKE", 99, "SCAM: Govt never asks Aadhaar/Bank on WhatsApp"
    if "moon" in t and "disappear" in t:
        return "FAKE", 99, "Science Fake: Moon cannot disappear"
    if "government" in t and "free" in t and "whatsapp" in t:
        return "FAKE", 98, "Fake Govt scheme on WhatsApp"
    # For ALL other fake - check sensational words
    if any(w in t for w in ["shocking","you wont believe","secret revealed","must share","viral"]):
        return "FAKE", 90, "Sensational / Clickbait without source"
    if "cures cancer" in t or "cures diabetes" in t:
        return "FAKE", 95, "Fake Health Claim"
    if "you have won" in t and "lottery" in t:
        return "FAKE", 97, "Lottery Scam"
    return "REAL", 85, "No fake pattern found"

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/check', methods=['POST'])
def check():
    try:
        txt = request.form.get('news_text','').strip()
        img_txt=""
        if 'image' in request.files:
            f=request.files['image']
            if f and f.filename!='':
                img=Image.open(f.stream).convert('L')
                img_txt=pytesseract.image_to_string(img, lang='eng')
        final=txt if txt else img_txt
        if not final.strip():
            return jsonify({'status':'error','message':'No text'})
        label,conf,reason=detect_fake_news(final)
        return jsonify({'status':'success','label':label,'confidence':conf,'reason':reason,'extracted_text':final[:500]})
    except Exception as e:
        return jsonify({'status':'error','message':str(e)})
if __name__=='__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',10000)))