from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import os, re

class TextEmbedding:
    def __init__(self):
        self.model = SentenceTransformer(os.path.join(Path(__file__).parent.parent, 'models','sentence_transformer'))
    
    # Clean text before getting embedding
    def clean_text(self, text):
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    # Get text embedding after cleaning it
    def get_text_embedding(self, text_list):
        cleaned_text_list = [self.clean_text(text) for text in text_list]
        return self.model.encode(cleaned_text_list)
    
    # Saving the embedding of collection of text
    def save_text_embedding(self, text_list, file_name):
        embeddings = self.get_text_embedding(text_list)
        np.save(file_name, embeddings)
