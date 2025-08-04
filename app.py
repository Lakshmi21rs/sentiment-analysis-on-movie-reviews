import streamlit as st
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import predict_sentiment  # ✅ Now works properly

# Streamlit UI
st.set_page_config(page_title="Movie Review Sentiment Analyzer")
st.title("🎬 Movie Review Sentiment Analyzer")

review = st.text_area("Enter your movie review:")

if st.button("Analyze"):
    if review.strip():
        prediction = predict_sentiment(review)
        st.success(f"Predicted Sentiment: **{prediction.upper()}**")
    else:
        st.warning("Please enter a valid review.")
