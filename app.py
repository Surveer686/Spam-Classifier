import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -----------------------------
# Download required NLTK data
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# -----------------------------
# Text preprocessing function
# -----------------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))

    return " ".join(y)

# -----------------------------
# Load trained model & vectorizer
# -----------------------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


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
