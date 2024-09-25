import faiss
import numpy as np

class SimilaritySearch:

    # read embeddings from file
    def __init__(self, file_name):
        self.embeddings = np.load(file_name)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    # search for similar embeddings
    def search(self, query_embedding, k):
        D, I = self.index.search(np.array([query_embedding]), k)
        return D, I
    
    # search for similar text
