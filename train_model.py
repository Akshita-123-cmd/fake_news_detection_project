import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------------
# 1. Load Dataset
# -----------------------------------

data = pd.read_csv("news.csv")

print("Dataset loaded successfully!")

print(data.head())


# -----------------------------------
# 2. Remove Missing Values
# -----------------------------------

data = data.dropna()


# -----------------------------------
# 3. Input and Output
# -----------------------------------

X = data["text"]
y = data["label"]


# -----------------------------------
# 4. Split Dataset
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------------
# 5. TF-IDF Vectorization
# -----------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


# -----------------------------------
# 6. Train ML Model
# -----------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)


# -----------------------------------
# 7. Test Model
# -----------------------------------

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")

print(classification_report(y_test, y_pred))


# -----------------------------------
# 8. Save Model
# -----------------------------------

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)


# -----------------------------------
# 9. Save Vectorizer
# -----------------------------------

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


print("\nModel saved successfully!")

print("vectorizer.pkl created successfully!")