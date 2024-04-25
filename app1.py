import streamlit as st
import pickle
import numpy as np
from scipy.sparse import csr_matrix
# Load models and vectorizer
with open('model_sp.pkl', 'rb') as f:
    model_sp = pickle.load(f)

with open('model_th_svc.pkl', 'rb') as f:
    model_th_svc = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('vectorizer_th.pkl', 'rb') as f:
    vectorizer2 = pickle.load(f)

def contains_abusive_language(text):
    abusive_words = ["kill", "bastard", "hate you", "hate", "murder", "blackmail", "threat", "warn", "rape", "slaughter", "abuse", "slam", "death", "hurt", "harm"]
    # Check if any abusive word is present in the text
    for word in abusive_words:
        if word in text.lower():
            return True
    return False

def preprocess_text_spam(text):
    text = vectorizer.transform([text])
    return text
def preprocess_text_th(text):
    text = vectorizer2.transform([text])
    return text
def predict_spam(text):
    prediction = model_sp.predict(text)
    return prediction[0]

def predict_threat(text):
    # Predict threat
    isp = csr_matrix(text , shape = (1,26800))
    prediction = model_th_svc.predict(isp)
    return prediction[0]


def main():
    st.title("Email Classification")
    email = st.text_area("Enter your email here:")
    if st.button("Classify"):
        if email.strip() == "":
            st.warning("Please enter an email.")
        else:
            processed_email_spam = preprocess_text_spam(email)
            spam_prediction = predict_spam(processed_email_spam)
            processed_email_th = preprocess_text_th(email)
            threat_prediction = predict_threat(processed_email_th)

            if spam_prediction == 1:
                st.error("It is a spam mail.")
            else:
                if contains_abusive_language(email):
                    st.error("It is not spam but a threat.")
                else:
                    st.success("It is not spam and not a threat.")
                    
if __name__ == "__main__":
    main()
