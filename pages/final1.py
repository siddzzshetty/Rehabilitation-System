import streamlit as st
import cv2
import os
import nest_asyncio
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage
import mediapipe as mp
import numpy as np
import torch
from collections import deque
from sklearn.preprocessing import StandardScaler
import mediapipe as mp 
from sklearn.preprocessing import MinMaxScaler
import pickle
import time
import threading
import torch.nn as nn
from st_helper import get_exercise_instructions, get_angles_to_calculate, display_sidebar, run_camera_feed
from chatbot_ui_final1 import chatbot_ui
from config import GROQ_API_KEY, chat_model, exercises

# Set up Streamlit Page Configuration
st.set_page_config(page_title="Fitness Exercise Assistant", layout="wide")

st.markdown(
    """
    <style>
        .pose-box {
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: #333;
            border: 2px solid #ccc;
        }

        .pose-true {
            background-color: #d4edda;
            color: #155724;
            border: 2px solid #c3e6cb;
        }

        .pose-false {
            background-color: #f8d7da;
            color: #721c24;
            border: 2px solid #f5c6cb;
        }
    </style>
    """,
    unsafe_allow_html=True
)

display_sidebar()

# @st.dialog("📌 Instructions")
# def get_instructions():
#     exercise = "squat"
#     for step in get_exercise_instructions(exercise):
#         st.write(f"✅ {step}")
#     if st.button("OK"):
#         st.session_state.show_popup = False
#         st.session_state.camera_started = True
#         # st.rerun()


# st.title(f"{exercise.title()} Exercise")
# st.header("📷 Live Camera Feed")

# Instruction + Pose Box Layout
col1, col2 = st.columns([5, 2])  # Adjust column ratio as needed

with col1:
    st.header("📷 Live Camera Feed")
    st.markdown("**📌 Position yourself 2m away from the device.**")  # Permanent instruction
    
    if "camera_started" not in st.session_state:
        st.session_state.camera_started = False

    if not st.session_state.camera_started:
        if st.button("📷 Start Camera", key="start_camera_button"):
            st.session_state.camera_started = True

    if st.session_state.camera_started:
        run_camera_feed(st.empty())  # Ensure camera feed stays in place

with col2:
    st.markdown("**🧍 Pose**")  # Pose title
    pose_placeholder = st.empty()
    # pose_placeholder.write("Waiting for pose...")  # No HTML, pure Streamlit

 # Styled Chatbot Box
    with st.container(border=True):
        chatbot_ui()  # Call chatbot function
    

# Camera start button with popup
if "camera_started" not in st.session_state:
    st.session_state.camera_started = False
if "show_popup" not in st.session_state:
    st.session_state.show_popup = False



# Run camera feed
if st.session_state.camera_started:
    run_camera_feed(pose_placeholder)