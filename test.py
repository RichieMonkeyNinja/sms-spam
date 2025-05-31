import streamlit as st
import joblib
import pandas as pd
from Function import extract_cls_embeddings, preprocess  # Your CLS embedding function

# Load your spam classifier model
lr_model = joblib.load("best_lr_model.pkl")
mnb_model = joblib.load('best_mnb.pkl')

vectorizer = joblib.load('tfidf_vectorizer.pkl')

prompt = 'Call this number for free tokens now!'
        # Extract CLS embedding and TF-IDF vector
        # embedding = extract_cls_embeddings(prompt)
prompt = 'Want explicit SEX in 30 secs? Ring 02073162414...'
clean_prompt = preprocess(prompt)
vectors = vectorizer.transform([clean_prompt])


# Predict using your model
# prediction_bert = lr_model.predict(embedding)[0]
# confidence_bert = lr_model.predict_proba(embedding)[0][prediction_bert] * 100
# label_bert = "SPAM" if prediction_bert == 1 else "HAM"

prediction_tfidf = mnb_model.predict(vectors)[0]
confidence_tfidf = mnb_model.predict_proba(vectors)[0][prediction_tfidf] * 100
label_tfidf = "SPAM" if prediction_tfidf == 1 else "HAM"

# Response with confidence
# response_bert = f"This message is classified as **{label_bert}** with {confidence_bert:.2f}% confidence by BERT + Logistic Regression."
response_tfidf = f"This message is classified as **{label_tfidf}** with {confidence_tfidf:.2f}% confidence by TF-IDF + SVM."
print(response_tfidf)
        # , use_container_width=True, height = 200)
