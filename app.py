from flask import Flask, render_template, request
import pickle


# -----------------------------------
# Create Flask Application
# -----------------------------------

app = Flask(__name__)


# -----------------------------------
# Load Trained Model
# -----------------------------------

with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# -----------------------------------
# Load TF-IDF Vectorizer
# -----------------------------------

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# -----------------------------------
# Home Page
# -----------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------------
# Fake News Prediction
# -----------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    news_text = request.form["news"]


    # Convert text into TF-IDF
    news_vector = vectorizer.transform([news_text])


    # Make prediction
    prediction = model.predict(news_vector)[0]


   

    # Send result to website
    return render_template(
        "index.html",
        prediction=prediction,
        
        news=news_text
    )


# -----------------------------------
# Run Application
# -----------------------------------

if __name__ == "__main__":
    app.run(debug=True)