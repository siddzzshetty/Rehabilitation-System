import os 
from langchain_groq import ChatGroq

GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
chat_model = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
USE_MODEL = True
exercises = {
    "🏋️ Squat": "squat",
    "🧎 Sit Up": "situp",
    "👐 Push-up": "pushup",
    "💪 Pull-up": "pullup",
    "🏃 Jumping Jacks": "jumpingjacks"
}

