import streamlit as st
import pickle
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizer
model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Function to preprocess text
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

# Streamlit UI
st.title("📱 SMS Spam Classifier")

message = st.text_area("Enter your message:")

if st.button("Predict"):
    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        st.error("This is a SPAM message!")
    else:
        st.success("This is a HAM (not spam) message.")

