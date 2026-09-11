from flask import Flask, render_template, request
import pickle
from PIL import Image
import pytesseract

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = ""

    # 1. Check if user uploaded an image
    if 'news_image' in request.files and request.files['news_image'].filename!= '':
        try:
            image_file = request.files['news_image']
            img = Image.open(image_file)
            # Extract text from image
            news_text = pytesseract.image_to_string(img)
            print(f"Text from image: {news_text}")
        except Exception as e:
            news_text = ""
            print(f"Image error: {e}")

    # 2. If no image text, use typed text
    if not news_text or news_text.strip() == "":
        news_text = request.form.get('news', '')

    # 3. If still empty, show error
    if not news_text or news_text.strip() == "":
        return render_template('index.html', prediction="Please enter news text or upload an image", news="")

    news_vector = vectorizer.transform([news_text])
    prediction = model.predict(news_vector)[0]

    return render_template('index.html', prediction=prediction, news=news_text)

if __name__ == '__main__':
    app.run(debug=True)