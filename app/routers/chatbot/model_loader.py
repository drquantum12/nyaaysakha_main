import os
from configparser import ConfigParser
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from .cloud_utility import download_from_gcs

config = ConfigParser()
config.read(os.environ["config_path"])

ml_models = dict()

def load_models():
    # download model weights if absent
    download_from_gcs(config.get("cloud_params", "bucket_name"))

    vector_store_path = config.get("data", "vector_store_path")
    ml_models["embed_model"] = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-mpnet-base-v2")
    ml_models["docstore"] = InMemoryDocstore()
    ml_models["vector_store"] = FAISS.load_local(vector_store_path, embeddings=ml_models["embed_model"], allow_dangerous_deserialization=True)

def clear_models():
    ml_models.clear()
    print("Models cleared.")