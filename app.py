import streamlit as st
import pickle

# Load the trained Naïve Bayes model
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("📩 Email Spam Classifier")

# User input
email_text = st.text_area("Enter your email content:")

# Classification button
if st.button("Classify"):
    if not email_text.strip():
        st.warning("Please enter some text to classify.")
    else:
        # Predict using the trained model
        prediction = model.predict([email_text])[0]

        # Display result
        if prediction == 1:
            st.error("🚨 SPAM")
        else:
            st.success("✅ HAM (Not Spam)")