from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from .dependencies import get_query_token
from .routers import chat, auth
from .routers.chatbot.model_loader import load_models, clear_models


@asynccontextmanager
async def lifespan(app:FastAPI):
    print("Start of lifespan")
    load_models()
    yield
    clear_models()
    print("End of lifespan")


app = FastAPI(lifespan=lifespan, dependencies=[Depends(get_query_token)])
app.include_router(auth.router)
app.include_router(chat.router)

@app.get("/")
def root():
    return {"msg": "Hello this is a rag application endpoint!"}