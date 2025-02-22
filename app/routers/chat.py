from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from .chatbot.chat_utility import ChatUtility
from ..dependencies import get_token_header
from .chatbot.model_loader import ml_models

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    dependencies=[Depends(get_token_header)],
    responses={404: {"description": "Not found"}},
)

class UserInput(BaseModel):
    text: str

@router.get("/")
async def get_all_conversations():
    return {"msg": "Get all conversations"}

@router.get("/getConversation/{conversation_id}")
async def get_conversation(conversation_id: int):
    return {"msg": f"Get conversation with id {conversation_id}"}
    
@router.post("/startChat/{conversation_id}")
async def chat(conversation_id: int, user_input: UserInput, token: str = Depends(get_token_header)):
    chat_utility = ChatUtility(ml_models["vector_store"])
    response = chat_utility.chat({"question": user_input.text})
    if not response:
        raise HTTPException(status_code=400, detail="No response from chat utility.")
    return {"results": response}