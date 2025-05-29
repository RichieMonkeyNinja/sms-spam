import streamlit as st
import joblib
import pandas as pd
from Function import extract_cls_embeddings  # Your CLS embedding function

# Sidebar: External links + classification history
with st.sidebar:
    st.markdown("[Learn more about BERT here](https://en.wikipedia.org/wiki/BERT_(language_model))")
    st.markdown("[Learn more about TF-IDF here](https://en.wikipedia.org/wiki/Tf%E2%80%93idf#:~:text=In%20information%20retrieval%2C%20tf%E2%80%93idf,appear%20more%20frequently%20in%20general.)")
    st.markdown("[View the source code](https://github.com/RichieMonkeyNinja)")
    st.markdown("---")
    st.markdown("**Classification History (Last 5)**")

# Load your spam classifier model
model = joblib.load("best_lr_model.pkl")

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
        # Extract CLS embedding
        embedding = extract_cls_embeddings(prompt)

        # Predict using your model
        prediction = model.predict(embedding)[0]
        confidence = model.predict_proba(embedding)[0][prediction] * 100
        label = "SPAM" if prediction == 1 else "HAM"

        # Response with confidence
        response = f"This message is classified as **{label}** with {confidence:.2f}% confidence."

        # , use_container_width=True, height = 200)

    except Exception as e:
        response = f"Error during classification: {e}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)