from contextlib import asynccontextmanager
from fastapi import FastAPI
from utility.vector_store import FaissVectorStore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from configparser import ConfigParser
from pydantic import BaseModel
from cloud_utility import download_from_gcs
import os

config = ConfigParser()
config.read(os.environ["config_path"])

ml_models = dict()

@asynccontextmanager
async def lifespan(app:FastAPI):
    print("Start of lifespan")
    
    # download model weights if absent
    download_from_gcs(config.get("cloud_params", "bucket_name"))

    vector_store_path = config.get("data", "vector_store_path")
    ml_models["embed_model"] = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-mpnet-base-v2")
    ml_models["docstore"] = InMemoryDocstore()
    ml_models["vector_store"] = FAISS.load_local(vector_store_path, embeddings=ml_models["embed_model"], allow_dangerous_deserialization=True)
    yield
    ml_models.clear()
    print("End of lifespan")


app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"msg": "Hello this is a rag application endpoint!"}


class UserInput(BaseModel):
    text:str

@app.post("/rag/retrieve")
async def retrieve(user_input:UserInput):
    results = ml_models["vector_store"].similarity_search(user_input.text, 5)
    return {"results": results}