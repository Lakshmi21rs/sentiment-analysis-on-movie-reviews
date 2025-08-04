import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from .preprocess import clean_text

def train_and_save_model():
    df = pd.read_csv("data/imdb_dataset.csv")
    df['review'] = df['review'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    joblib.dump(model, "models/sentiment_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")
