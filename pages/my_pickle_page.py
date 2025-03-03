import streamlit as st
import pickle
from config import exercises

st.title("This is just a demo file 🎀")
def show_sidebar():
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    if st.sidebar.button("📊 View Graphs"):
        st.session_state.page = "graphs"
        st.rerun()

    if st.sidebar.button("💬 Chatbot"):
        st.switch_page("pages/chatbot_page.py")

    if st.sidebar.button("pickle_page"):
        st.switch_page("pages/my_pickle_page.py")

    st.sidebar.subheader("Select Exercise")

    for label, key in exercises.items():
        if st.sidebar.button(label):
            st.session_state.exercise = key
            st.rerun()

show_sidebar()
def save_pickle_file(pose):
    with open("data.pkl", "wb") as f:
        pickle.dump(pose, f)

def load_pickle_file():
    with open("data.pkl", "rb") as f:
        pose = pickle.load(f)
        st.write(pose)

col1, col2 = st.columns([1, 1])  # Adjust column ratio as needed

with col1:
    if st.button("True"):
        pose = True
        save_pickle_file(pose)
        load_pickle_file()

with col2:
    if st.button("False"):
        pose = False
        save_pickle_file(pose)
        load_pickle_file()
        
