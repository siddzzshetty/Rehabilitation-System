import streamlit as st
import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage
from config import exercises
from st_helper import chat_with_exercise_assistant, display_sidebar

st.set_page_config(page_title="Fitness Exercise Assistant", layout="wide")

display_sidebar()

st.title("💬 AI Chatbot")
st.subheader("Chat with your AI fitness coach!")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm your AI coach. How can I help with your exercise?")
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

    # Get AI response
    response = chat_with_exercise_assistant(st.session_state.chat_history, user_input)

    # Display AI response
    with st.chat_message("assistant"):
        st.write(response)

    # Store messages in history
    st.session_state.chat_history.append(AIMessage(content=response))