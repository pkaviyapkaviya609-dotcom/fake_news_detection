"""
app.py
------
Streamlit web app for the Fake News Detector.

Run with:
    streamlit run app.py

Requires fake_news_model.pkl and tfidf_vectorizer.pkl
(created by running train_model.py first) to be in the same folder.
"""

import re
import string
import joblib
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)

    words = text.split()
    words = [STEMMER.stem(w) for w in words if w not in STOPWORDS]
    return " ".join(words)


@st.cache_resource
def load_artifacts():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


def main():
    st.set_page_config(page_title="Fake News Detector", page_icon="📰")
    st.title("📰 Fake News Detector")
    st.write(
        "Paste a news headline or article below and the model will predict "
        "whether it looks like **Real** or **Fake** news, based on writing "
        "patterns learned from a labeled training dataset."
    )

    model, vectorizer = load_artifacts()

    user_input = st.text_area("Enter news text here:", height=200)

    if st.button("Predict"):
        if not user_input.strip():
            st.warning("Please enter some text first.")
            return

        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])

        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        label = "🟥 Fake News" if prediction == 1 else "🟩 Real News"
        confidence = proba[prediction] * 100

        st.subheader(label)
        st.write(f"Confidence: **{confidence:.2f}%**")

        st.progress(int(confidence))

        st.caption(
            "Note: this model detects statistical writing patterns associated "
            "with fake news in the training data — it does not verify facts "
            "against real-world sources."
        )


if __name__ == "__main__":
    main()
