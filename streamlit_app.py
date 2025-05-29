import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Import CLS Embeddings Extraction Function
from Function import extract_cls_embeddings

# Load the trained model
# Custom background style
st.markdown("""
    <style>
    body {
        background-color: #e6e6e6;
    }
    </style>
    """, unsafe_allow_html=True)

st.spinner("Loading BERT... please wait")
st.title('Spam Classifier using BERT + Logistic Regression')
st.write('Enter a message to see the classification in action')
st.image("spam_image.png", width=120, caption="An example image")


# Initialize classification history
if 'history' not in st.session_state:
    st.session_state['history'] = pd.DataFrame(columns=['Message', 'Prediction'])

# User input
user_input = st.text_area('Message', height=150)

# Predict button
if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter a message to classify.")
    else:
        try:
            embedding = extract_cls_embeddings(user_input)
            prediction = model.predict(embedding)
            label = "SPAM" if prediction == 1 else "HAM"

            # Append to history (limit to top 5 latest)
            new_row = pd.DataFrame({'Message': [user_input], 'Prediction': [label]})
            st.session_state['history'] = pd.concat([new_row, st.session_state['history']]).head(5)

            st.success(f"Prediction: {label}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Display history
st.subheader("Classification History (Last 5)")
st.dataframe(st.session_state['history'], use_container_width=True)
