# from langchain.chat_models import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI 


load_dotenv()  # Load GOOGLE_API_KEY from .env

def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
