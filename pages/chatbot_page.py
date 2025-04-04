import streamlit as st
import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from config import exercises
from st_helper import display_sidebar

st.set_page_config(page_title="Fitness Exercise Assistant", layout="wide")

display_sidebar()

st.title("💬 AI Chatbot")
st.subheader("Chat with your AI fitness coach!")

prompt_template = ChatPromptTemplate([
    ("system",  "You are a responsible AI medical chatbot.  You must follow these rules strictly: Do NOT provide medical advice, exercise recommendations, or plans.Only answer general doubts about fitness, how to do exercises  or explain general terms. If a user asks for exercise suggestions, ALWAYS reply: 'I cannot provide exercise recommendations. Please consult a doctor or a physiotherapist for advice.'"),
    ("user", "{question}")
])

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm your AI assistant. I can help answer general questions about exercise. How can I assist you?")
    ]

# Display past messages in order
for message in st.session_state.chat_history:
    role = "assistant" if isinstance(message, AIMessage) else "user"
    with st.chat_message(role):
        st.write(message.content)

# User Input for Chatbot
user_input = st.chat_input("Type your question here...")

if user_input:
    # Append user message
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Generate AI response using the prompt
    format_prompt = prompt_template.format(question=user_input)
    response = ChatGroq(model="llama3-8b-8192").invoke([HumanMessage(content=format_prompt)]).content

    # Display AI response
    with st.chat_message("assistant"):
        st.write(response)

    # Store messages in history
    st.session_state.chat_history.append(AIMessage(content=response))
