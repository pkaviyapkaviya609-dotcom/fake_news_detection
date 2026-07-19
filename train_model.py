"""
train_model.py
----------------
Loads the Fake/Real news dataset, cleans the text, trains a
Logistic Regression classifier on TF-IDF features, evaluates it,
and saves the trained model + vectorizer to disk.

Before running:
1. Download "Fake.csv" and "True.csv" from the Kaggle dataset:
   https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Place both files in the same folder as this script.
3. Run: python train_model.py
"""

import re
import string
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# 1. Download NLTK resources (only needed once)
# ---------------------------------------------------------
nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


# ---------------------------------------------------------
# 2. Text cleaning function
# ---------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/numbers, remove stopwords, and stem."""
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)                  # remove text in brackets
    text = re.sub(r"https?://\S+|www\.\S+", "", text)    # remove URLs
    text = re.sub(r"<.*?>+", "", text)                   # remove HTML tags
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # remove punctuation
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)                 # remove words with numbers

    words = text.split()
    words = [STEMMER.stem(w) for w in words if w not in STOPWORDS]
    return " ".join(words)


# ---------------------------------------------------------
# 3. Load and prepare dataset
# ---------------------------------------------------------
def load_data():
    fake_df = pd.read_csv("Fake.csv")
    true_df = pd.read_csv("True.csv")

    fake_df["label"] = 1   # 1 = Fake
    true_df["label"] = 0   # 0 = Real

    df = pd.concat([fake_df, true_df], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle rows

    # Combine title + text into a single field
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["content"] = df["content"].apply(clean_text)

    return df[["content", "label"]]


# ---------------------------------------------------------
# 4. Main training pipeline
# ---------------------------------------------------------
def main():
    print("Loading and cleaning data...")
    df = load_data()

    X = df["content"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nSaving model and vectorizer...")
    joblib.dump(model, "fake_news_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("Done! Files saved: fake_news_model.pkl, tfidf_vectorizer.pkl")


if __name__ == "__main__":
    main()
