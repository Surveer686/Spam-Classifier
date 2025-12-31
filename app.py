import streamlit as st
import pickle
import nltk
import string
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# -----------------------------
# Cloud-safe NLTK setup (HF + Streamlit)
# -----------------------------
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

nltk.data.path.append(NLTK_DATA_DIR)

nltk.download("punkt", download_dir=NLTK_DATA_DIR)
nltk.download("stopwords", download_dir=NLTK_DATA_DIR)

# -----------------------------
# Initialize tools once
# -----------------------------
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# -----------------------------
# Text preprocessing function
# -----------------------------
def transform_text(text):
    text = text.lower()
    tokens = word_tokenize(text)

    cleaned_words = []
    for word in tokens:
        if word.isalnum() and word not in stop_words:
            cleaned_words.append(ps.stem(word))

    return " ".join(cleaned_words)

# -----------------------------
# Load trained model & vectorizer
# -----------------------------
with open("vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📩 SMS Spam Classifier")
st.write("Enter an SMS message to check whether it is **Spam** or **Ham**.")

input_sms = st.text_area("Enter the message")

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT Spam")
