# Kernel 1: Import Libraries
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

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