import streamlit as st
import pickle
from config import exercises
from st_helper import display_sidebar

st.title("This is just a demo file 🎀")
display_sidebar()

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
        
