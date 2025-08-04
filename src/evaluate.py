import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
from .preprocess import clean_text

def evaluate_model():
    df = pd.read_csv("data/imdb_dataset.csv")
    df['review'] = df['review'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
