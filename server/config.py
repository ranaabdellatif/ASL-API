import os
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class Config:
    MONGO_URI = os.getenv("MONGO_URI")
    SECRET_KEY = os.getenv("JWT_SECRET_KEY")
