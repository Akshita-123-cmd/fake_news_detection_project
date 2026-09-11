import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
texts = ["this is real news", "this is fake news", "breaking real story", "fake clickbait story"]
labels = ["REAL", "FAKE", "REAL", "FAKE"]
vec = TfidfVectorizer()
X = vec.fit_transform(texts)
model = PassiveAggressiveClassifier()
model.fit(X, labels)
pickle.dump(model, open('model.pkl','wb'))
pickle.dump(vec, open('vectorizer.pkl','wb'))
print("created")