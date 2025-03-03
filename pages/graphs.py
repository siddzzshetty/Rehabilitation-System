import streamlit as st
from config import exercises
from st_helper import display_sidebar

st.set_page_config(page_title="Fitness Exercise Assistant", layout="wide")

display_sidebar()


st.title("📈 Performance Graphs")
st.write("Your exercise performance graphs will appear here.")
st.line_chart({"Squats": [10, 12, 15, 20], "Push-ups": [5, 8, 10, 12]})