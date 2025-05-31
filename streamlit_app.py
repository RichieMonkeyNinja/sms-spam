import streamlit as st
import joblib
import pandas as pd
from Function import extract_cls_embeddings  # Your CLS embedding function
from Function import preprocess
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Sidebar: External links + classification history
with st.sidebar:
    st.markdown("[Learn more about BERT here](https://en.wikipedia.org/wiki/BERT_(language_model))")
    st.markdown("[Learn more about TF-IDF here](https://en.wikipedia.org/wiki/Tf%E2%80%93idf#:~:text=In%20information%20retrieval%2C%20tf%E2%80%93idf,appear%20more%20frequently%20in%20general.)")
    st.markdown("[View the source code](https://github.com/RichieMonkeyNinja)")
    st.markdown("---")
    st.markdown("**Classification History (Last 5)**")

# Load your spam classifier model
lr_model = joblib.load("best_lr_model.pkl")
mnb_model = joblib.load('best_mnb.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize chat and classification history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Paste an SMS and I'll classify it as SPAM or HAM."}]

# if "history_table" not in st.session_state:
#     st.session_state["history_table"] = pd.DataFrame(columns=["Message", "Label", "Confidence (%)"])
    
# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
if prompt := st.chat_input("Type your SMS here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    try:
        # Extract CLS embedding and TF-IDF vector
        embedding = extract_cls_embeddings(prompt)
        clean_sms = preprocess(prompt)
        vectors = vectorizer.transform([clean_sms])

        # Predict using your model
        prediction_bert = lr_model.predict(embedding)[0]
        confidence_bert = lr_model.predict_proba(embedding)[0][prediction_bert] * 100
        label_bert = "SPAM" if prediction_bert == 1 else "HAM"

        prediction_tfidf = mnb_model.predict(vectors)[0]
        confidence_tfidf = mnb_model.predict_proba(vectors)[0][prediction_tfidf] * 100
        label_tfidf = "SPAM" if prediction_tfidf == 1 else "HAM"

        # Response with confidence
        response_bert = f"This message is classified as **{label_bert}** with {confidence_bert:.2f}% confidence by BERT + Logistic Regression."
        response_tfidf = f"This message is classified as **{label_tfidf}** with {confidence_tfidf:.2f}% confidence by TF-IDF + SVM."
        # , use_container_width=True, height = 200)

    except Exception as e:
        response_bert = f"Error during classification: {e}"

    st.session_state.messages.append({"role": "assistant", "content": response_bert, 'name':'BERT Assistant'})
    st.session_state.messages.append({"role": "assistant", "content": response_tfidf, 'name':'TF-IDF Assistant'})    
    st.chat_message("assistant").write(response_bert)
    st.chat_message("assistant").write(response_tfidf)    