from google.cloud import firestore
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
import os, uuid
from fastapi import APIRouter, HTTPException

router = APIRouter(
    prefix="/auth",
    tags=["auth"],
    responses={404: {"description": "Not found"}},
)

firestore_client = firestore.Client.from_service_account_json(os.environ["GOOGLE_FIRESTORE_CREDENTIALS"])

users_ref = firestore_client.collection("users")

class UserCreate(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.post("/signup/", tags=["auth"])
async def signup(user: UserCreate):
    existing_user = users_ref.where("email", "==", user.email).stream()
    if any(existing_user):
        raise HTTPException(status_code=400, detail="User already exists")
    
    hashed_password = pwd_context.hash(user.password)
    user_id = str(uuid.uuid4())
    users_ref.document(user_id).set({
        "first_name": user.first_name,
        "last_name": user.last_name,
        "email": user.email,
        "password": hashed_password
    })
    return {"msg": "User registered successfully", "user_id": user_id}

@router.post("/login/", tags=["auth"])
async def login(user: UserLogin):
    user_docs = users_ref.where("email", "==", user.email).stream()
    user_data = None
    for doc in user_docs:
        user_data = doc.to_dict()
        break
    if not user_data or not pwd_context.verify(user.password, user_data["password"]):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    return {"msg": "Login successful"}