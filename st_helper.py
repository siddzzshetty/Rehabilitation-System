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
from config import chat_model, exercises

def display_sidebar():
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    if st.sidebar.button("📊 View Graphs"):
        st.switch_page("pages/graphs.py")

    if st.sidebar.button("💬 Chatbot"):
        st.switch_page("pages/chatbot_page.py")

    if st.sidebar.button("pickle_page"):
        st.switch_page("pages/my_pickle_page.py")

    st.sidebar.subheader("Select Exercise")

    for label, key in exercises.items():
        if st.sidebar.button(label):
            st.session_state.exercise = key
            st.rerun()

# # Function to display Graphs
# def display_graphs():
#     st.title("📈 Performance Graphs")
#     st.write("Your exercise performance graphs will appear here.")
#     st.line_chart({"Squats": [10, 12, 15, 20], "Push-ups": [5, 8, 10, 12]})


# # Function to handle AI Chatbot
# def display_chatbot():
#     st.title("💬 AI Chatbot")
#     st.subheader("Chat with your AI fitness coach!")

#     # Display past messages in order
#     for message in st.session_state.chat_history:
#         role = "assistant" if isinstance(message, AIMessage) else "user"
#         with st.chat_message(role):
#             st.write(message.content)

#     # User Input for Chatbot
#     user_input = st.chat_input("Type your question here...")

#     if user_input:
#         # Append user message
#         st.session_state.chat_history.append(HumanMessage(content=user_input))

#         # Display user message
#         with st.chat_message("user"):
#             st.write(user_input)

#         # Get AI response
#         response = chat_with_exercise_assistant(st.session_state.chat_history, user_input)

#         # Display AI response
#         with st.chat_message("assistant"):
#             st.write(response)

#         # Store messages in history
#         st.session_state.chat_history.append(AIMessage(content=response))


# Function to chat with AI assistant
def chat_with_exercise_assistant(history, user_input):
    # Keywords to detect restricted queries
    restricted_keywords = ["medicine", "prescription", "exercise plan", "workout plan", "treatment", 
                           "therapy", "rehabilitation", "physical therapy", "physiotherapy", "routine", 
                           "recovery exercises", "fitness schedule", "training program"]

    user_input_lower = user_input.lower()

    # Check for restricted keywords
    if any(keyword in user_input_lower for keyword in restricted_keywords):
        response = "I'm not a medical professional. Please contact a physiotherapist for proper guidance."
    else:
        # Only send the latest message, not the entire history
        ai_response = chat_model.invoke([HumanMessage(content=user_input)])

        response = ai_response.content

    return response

# Function to show popup with exercise instructions
def show_instructions_popup(exercise):
    st.session_state.show_popup = True

# Function to get exercise instructions
def get_exercise_instructions(exercise):
    instructions = {
        "squat": ["Stand with feet shoulder-width apart.", "Keep your back straight.", "Lower your body by bending your knees.", "Return to the starting position."],
        "situp": ["Lie on your back with knees bent.", "Place hands behind your head.", "Lift your upper body towards your knees.", "Lower back down with control."],
        "pushup": ["Start in a plank position.", "Lower your body until your chest nearly touches the floor.", "Push back up to starting position.", "Keep your body straight."],
        "pullup": ["Hang from a bar with palms facing away.", "Pull yourself up until your chin is over the bar.", "Lower yourself with control.", "Repeat while maintaining form."],
        "jumpingjacks": ["Stand with feet together.", "Jump and spread legs while raising arms.", "Jump back to starting position.", "Maintain a steady rhythm."]
    }
    return instructions.get(exercise, ["No instructions available"])


def get_angles_to_calculate(landmarks, mp_pose):
    # Define angles to calculate
    angles_to_calculate = {
        "right_elbow_right_shoulder_right_hip": [
            [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
            [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
            [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
        ],
        "left_elbow_left_shoulder_left_hip": [
            [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
        ],
        "right_knee_mid_hip_left_knee": [
            [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
            [(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x + landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x) / 2,
            (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y + landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y) / 2],
            [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
        ],
        "right_hip_right_knee_right_ankle": [
            [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
            [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
            [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y],
        ],
        "left_hip_left_knee_left_ankle": [
            [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y],
        ],
        "right_wrist_right_elbow_right_shoulder": [
            [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y],
            [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
            [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
        ],
        "left_wrist_left_elbow_left_shoulder": [
            [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
            [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        ],
    }

    return angles_to_calculate