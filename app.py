from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']
    news_vector = vectorizer.transform([news_text])
    prediction = model.predict(news_vector)[0]

    return render_template('index.html', prediction=prediction, news=news_text)

if __name__ == '__main__':
    app.run()