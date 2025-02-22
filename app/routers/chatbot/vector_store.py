from langchain.embeddings import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from uuid import uuid4
from configparser import ConfigParser
import pandas as pd

config = ConfigParser()
config.read("config.ini")

class FaissVectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings("sentence-transformers/paraphrase-mpnet-base-v2")
        self.docstore = InMemoryDocstore()
        self.index = faiss.IndexFlatL2(len(self.embeddings.embed_query("test")))
        self.source_data_file_path = config.get("data", "source_data_file_path")
        self.vector_store_path = config.get("data", "vector_store_path")

    def load_source_data(self):
        documents = list()
        records = pd.read_csv(self.source_data_file_path).to_dict(orient="records")
        for record in records:
            doc = Document(
                content=record["filtered_desc"],
                metadata={"scheme_type": record["scheme_type"],
                           "scheme_name": record["title"],
                           "more_detail_link": record["more_detail_link"],
                           "external_link": record["ext_link"],
                           },
            )
            documents.append(doc)
        return documents
    
    def build_and_save_vector_store(self):
        documents = self.load_source_data()
        uuids = [str(uuid4()) for _ in range(len(documents))]

        vector_store = FAISS(
            embedding_function=self.embeddings,
            index=self.index,
            docstore=self.docstore,
            index_to_docstore_id={},
        )

        vector_store.add_documents(documents=documents, ids=uuids)
        vector_store.save_local(self.vector_store_path)
    
    def load_vector_store(self):
        vector_store = FAISS.load_local(self.vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
        return vector_store

