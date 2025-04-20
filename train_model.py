import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download("stopwords")

# ✅ Load the dataset (Replace with your dataset if needed)
data = pd.read_csv("IMDB Dataset.csv")

# ✅ Preprocessing
data["sentiment"] = data["sentiment"].map({"positive": 1, "negative": 0})  # Convert to 1 & 0

# ✅ Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(data["review"], data["sentiment"], test_size=0.2, random_state=42)

# ✅ Creating the pipeline (TF-IDF + Naive Bayes)
model = make_pipeline(TfidfVectorizer(stop_words=stopwords.words("english")), MultinomialNB())

# ✅ Train the model
model.fit(X_train, y_train)

# ✅ Evaluate model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# ✅ Save model using Pickle
with open("sentiment_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as 'sentiment_model.pkl'")
