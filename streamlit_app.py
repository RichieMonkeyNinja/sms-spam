import streamlit as st
import joblib 
import numpy as np, pandas as pd

# Import CLS Embeddings Extraction Function
from Function import extract_cls_embeddings
# Load the best trained model from the .pk file 
model = joblib.load('best_lr_model.pkl')

# Streamlit UI
st.markdown("""
    <style>
    body {
        background-color: #e6e6e6;
    }
    </style>
    """, unsafe_allow_html=True)

st.spinner("Loading BERT... please wait")
st.title('Spam Classifier using BERT + Logistic Regression')
st.write('Enter a message to see the **classification** in action')
st.image("spam_image.png", width=120, caption="An example image")

# User Input
user_input = st.text_area('Message', height = 150)


# Predict Button
if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter a message to classify.")
    else:
        try:
            # Extract [CLS] embedding
            embedding = extract_cls_embeddings(user_input)

            # Predict
            prediction = model.predict(embedding)
            label = "SPAM" if prediction == 1 else "HAM"
            st.success(f"Prediction: **{label}**")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

