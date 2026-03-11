import os 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

DEBUGGING = False
GROQ_API_KEY=os.environ["GROQ_API_KEY"] 
chat_model = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
USE_MODEL = True
exercises = {
    "🏋️ Squat": "squat",
    "🧎 Sit Up": "situp",
    "👐 Push-up": "pushup",
    "💪 Pull-up": "pullup",
    "🏃 Jumping Jacks": "jumpingjack"
}

prompt_template_pushup = ChatPromptTemplate([
        ("system", """the patient is doing pushups. you will be provided with elbow angles  of 3 frame runs,
         where each frame run is of 5 frames where each frame run indicates 5 timestamps. 
         dont tell about the predicted coordinates and frame runs or give the user the predicted coordinates, thats just for your reference.
         you have to tell the user what is wrong in their form and how can they fix it. keep your answer concise. you have to answer in the following format.
         NOTE: do not even give the angle values or frame values. focus on keeping answer as short as possible.
         Incorrect: what is wrong about my current form
         Suggestion: how can i improve my current form 
        """),
        ("user", "{question}")
    ])

prompt_template_squat = ChatPromptTemplate([
        ("system", """ you will be provided with angles of left knee and right knee of a patient undergoing physiotherapy. you will get angles of 3 frame runs,
         where each frame run is of 5 frames and 2 columns(left knee, right knee). the patient is doing squats.
         dont tell about the predicted coordinates and frame runs or give the user the predicted coordinates, thats just for your reference.
         you have to tell the user what is wrong in their form and how can they fix it. keep your answer concise. you have to answer in the following format.
         NOTE: do not even give the angle values or frame values. focus on keeping answer as short as possible.
         answer as if you are talking to a patient
         ignore inconsistencies between right and left knee angles
         Incorrect: what is wrong about my current form
         Suggestion: how can i improve my current form 
        """),
        ("user", "{question}")
    ])

llm_model = {"1":"llama-3.1-8b-instant",
             "2":"meta-llama/llama-4-maverick-17b-128e-instruct",
             "3": "gemma2-9b-it"}


llm = ChatGroq(model=llm_model.get("1"))