import gradio as gr
import pickle
import nltk
import os
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# -----------------------------
# Cloud-safe NLTK setup
# -----------------------------
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

nltk.download("punkt", download_dir=NLTK_DATA_DIR)
nltk.download("stopwords", download_dir=NLTK_DATA_DIR)

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# -----------------------------
# Load model & vectorizer
# -----------------------------
with open("vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Text preprocessing
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
# Prediction function
# -----------------------------
def predict_spam(message):
    if message.strip() == "":
        return "⚠️ Please enter a message"

    transformed_sms = transform_text(message)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        return "🚨 SPAM MESSAGE"
    else:
        return "✅ NOT SPAM"

# -----------------------------
# Gradio UI
# -----------------------------
demo = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(lines=4, placeholder="Enter SMS message here..."),
    outputs="text",
    title="📩 SMS Spam Classifier",
    description="NLP-based SMS Spam Classifier using Machine Learning"
)

demo.launch()
