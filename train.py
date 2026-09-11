import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

df = pd.read_csv('news.csv')
# change column names if different - check your news.csv
# assuming first column = text, second = label
X = df.iloc[:,0]
y = df.iloc[:,1]

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

model = PassiveAggressiveClassifier()
model.fit(X_vec, y)

pickle.dump(model, open('model.pkl','wb'))
pickle.dump(vectorizer, open('vectorizer.pkl','wb'))
print("Done - both files created")