import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -----------------------------
# Cloud-safe NLTK downloads
# -----------------------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

# -----------------------------
# Text preprocessing function
# -----------------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric tokens
    words = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuation, apply stemming
    cleaned_words = [
        ps.stem(word) 
        for word in words 
        if word not in stopwords.words('english') and word not in string.punctuation
    ]

    return " ".join(cleaned_words)

# -----------------------------
# Load trained model & vectorizer
# -----------------------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))   # Make sure this file exists
model = pickle.load(open('model.pkl', 'rb'))        # Make sure this file exists

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
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display result
        if result == 1:
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT Spam")
