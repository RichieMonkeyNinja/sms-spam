# Kernel 1: Import Libraries
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import nltk 
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab') 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

def extract_cls_embeddings(texts):
    embeddings = []
    for text in [texts]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_vec = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(cls_vec)
    return np.array(embeddings)

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())  # Lowercase and tokenize
    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words  # Keep only alphabetic tokens and remove stopwords
    ]
    return ' '.join(cleaned_tokens)