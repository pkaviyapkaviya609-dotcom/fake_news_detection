# Fake News Detector (AI Mini Project)

A machine learning project that classifies news articles as **Real** or **Fake**
using TF-IDF text features and a Logistic Regression classifier, with a
Streamlit web interface for live predictions.

## How it works

1. **Text cleaning** – lowercase, remove punctuation/numbers/URLs, remove
   stopwords, apply stemming.
2. **Feature extraction** – TF-IDF converts each article into a numeric vector
   representing important, distinctive words.
3. **Model training** – Logistic Regression learns a boundary between
   real-news and fake-news word patterns.
4. **Prediction** – new text is cleaned, vectorized, and classified, with a
   confidence score.

## Setup

```bash
pip install -r requirements.txt
```

1. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Place `Fake.csv` and `True.csv` in this folder.

## Train the model

```bash
python train_model.py
```

This prints accuracy/precision/recall and saves:
- `fake_news_model.pkl`
- `tfidf_vectorizer.pkl`

## Run the web app

```bash
streamlit run app.py
```

Opens a local browser page where you can paste an article and get a
Real/Fake prediction with a confidence score.

## Deploy (optional, for a live demo link)

1. Push this folder to a GitHub repository (include the `.pkl` files).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your
   GitHub repo, and deploy `app.py`.
3. You'll get a public URL to show in your project submission/demo.

## Limitations (good to mention in your report/viva)

This model detects **statistical writing-style patterns** correlated with
fake news in its training data — it does not fact-check claims against
real-world sources. It can be fooled by real news written in an unusual
style, or fake news that mimics real reporting style closely.
