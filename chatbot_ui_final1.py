import streamlit as st
import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from config import exercises

def chatbot_ui():
    prompt_template = ChatPromptTemplate([
        ("system", "You are an AI medical chatbot. You can answer general doubts about exercise but **cannot** provide any medical advice, exercise suggestions, exercise plans, or recommendations under any circumstances. Only a doctor can provide such guidance. If asked for specific exercises, always respond with: 'I cannot provide exercise recommendations. Please consult a doctor or a physiotherapist for advice.'"),
        ("user", "{question}")
    ])

    # Initialize chat history in session state
    if "frontpage_chat_history" not in st.session_state:
        st.session_state.frontpage_chat_history = [
            AIMessage(content="Hello! I'm your AI assistant. How can I assist you?")
        ]

    with st.container():
        st.markdown("REMARKS ⚠️")  # Chatbot title
        
        # Messages container (Scroll effect via max_items)
        messages_container = st.empty()

        # Display chat history
        messages_display = []
        for message in st.session_state.frontpage_chat_history:
            role = "assistant" if isinstance(message, AIMessage) else "user"
            messages_display.append(f"**{role.capitalize()}:** {message.content}")

        messages_container.write("\n\n".join(messages_display))

        user_input = st.chat_input("Type your question here...")
        # Process User Input
        if user_input:
            st.session_state.frontpage_chat_history.append(HumanMessage(content=user_input))

            # Generate AI response
            format_prompt = prompt_template.format(question=user_input)
            response = ChatGroq().invoke([HumanMessage(content=format_prompt)]).content

            # Append AI response
            st.session_state.frontpage_chat_history.append(AIMessage(content=response))


            # Refresh chat messages
            messages_display.append(f"**User:** {user_input}")
            messages_display.append(f"**Assistant:** {response}")
            messages_container.write("\n\n".join(messages_display))

            

