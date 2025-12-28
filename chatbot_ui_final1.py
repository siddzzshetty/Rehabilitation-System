import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from config import exercises

prompt_template = ChatPromptTemplate([
    ("system", """You are a responsible AI medical chatbot.  You must follow these rules strictly:
     Do NOT provide medical advice, exercise recommendations, or plans.
     Only answer general doubts about fitness, how to do exercises  or explain general terms. 
     If a user asks for exercise suggestions, ALWAYS reply: 'I cannot provide exercise recommendations. 
     Please consult a doctor or a physiotherapist for advice.'"""),
    ("user", "{question}")
])

def chatbot_ui(prompt_template=prompt_template):
    # Initialize chat history in session state
    if "frontpage_chat_history" not in st.session_state:
        st.session_state.frontpage_chat_history = [
            AIMessage(content="Hello! I'm your AI assistant. How can I assist you?")
        ]

    st.markdown("## REMARKS ⚠️")

    # Chat container with scrolling using Streamlit elements
    chat_container = st.container()
    with chat_container:
        st.write("### Chat History")
        chat_history_area = st.empty()

        # Reverse the messages to get latest ones first
        message_list = list(reversed(st.session_state.frontpage_chat_history))

        # Separate User and AI messages
        user_messages = [msg for msg in message_list if isinstance(msg, HumanMessage)]
        ai_messages = [msg for msg in message_list if isinstance(msg, AIMessage)]

        # Ensure we display User → AI in reversed order
        chat_html = "<div style='height: 400px; overflow-y: auto; display: flex; flex-direction: column; border: 1px solid #ccc; padding: 10px;'>"

        # Display User → AI pairs
        for i in range(min(len(user_messages), len(ai_messages))):
            chat_html += f"<p><b>Assistant:</b> {ai_messages[i].content}</p>"
            chat_html += f"<p><b>User:</b> {user_messages[i].content}</p>"

        # Display any extra AI message (e.g., initial greeting)
        if len(ai_messages) > len(user_messages):
            chat_html += f"<p><b>Assistant:</b> {ai_messages[len(user_messages)].content}</p>"

        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)

    # User input field fixed at bottom
    user_input = st.chat_input("Type your question here...")
    if user_input:
        st.session_state.frontpage_chat_history.append(HumanMessage(content=user_input))

        # Display loading state
        with st.spinner('Generating response...'):
            format_prompt = prompt_template.format(question=user_input)
            response = ChatGroq(model="llama3-8b-8192").invoke([HumanMessage(content=format_prompt)]).content

        # Append AI response
        st.session_state.frontpage_chat_history.append(AIMessage(content=response))
    
        # Force rerun to refresh chat
        st.rerun()

if __name__ == "__main__":
    chatbot_ui()