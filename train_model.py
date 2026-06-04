import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("news_dataset.csv")

# Features and labels
X = df["text"]
y = df["category"]

# Convert text to numbers
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vectorized, y)

# Save model and vectorizer
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")


print("Model trained successfully!")