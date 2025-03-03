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
from st_helper import show_instructions_popup, get_exercise_instructions, get_angles_to_calculate, display_sidebar
from config import GROQ_API_KEY, chat_model, exercises

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
INPUT_WINDOW= 50
OUTPUT_WINDOW = 20
PRED_FREQ = 20
pose = None
pose_sequences = deque(maxlen=INPUT_WINDOW)
real_time_storage = []
frame_count = 0
collecting_real_time = False #this is a flag (false-during prediction, true-calculating real time values)
predicted_vs_real_storage = [] 
focus_angles=[]
pose = None 
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_loss(predicted, actual):
    mae_loss = nn.MSELoss()
    loss = mae_loss(torch.tensor(predicted),torch.tensor(actual))
    return loss.item()

def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, bc = b - a, c - b
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def get_pose(pose_placeholder):
    """Update the displayed pose value dynamically with CSS classes."""
    # pose_placeholder = st.empty()
    global pose
    try:
        with open("data.pkl", "rb") as f:
            new_pose = pickle.load(f)
            if new_pose != pose:
                pose = new_pose
                pose_class = "pose-true" if pose else "pose-false"
                pose_placeholder.markdown(
                    f'<div class="pose-box {pose_class}">{str(pose).upper()}</div>',
                    unsafe_allow_html=True
                )

                print(pose)
    except Exception as e:
        print(f"Error loading pose: {e}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

lstm_model = torch.jit.load(r'model_my_squatss_scripted.pt')#for linux relative path
lstm_model.to(device)
lstm_model.eval()

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(np.array([[0] * 7, [180] * 7]))




# Allow Streamlit to run in Jupyter Notebook
nest_asyncio.apply()

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


# Function to display Exercise Page
def display_exercise_page(exercise):
    st.title(f"{exercise.title()} Exercise")
    st.header("📷 Live Camera Feed")

    # Instruction + Pose Box Layout
    col1, col2 = st.columns([3, 1])  # Adjust column ratio as needed

    with col1:
        st.markdown("**📌 Position yourself 2m away from the device.**")  # Permanent instruction

    with col2:
        st.markdown("**🧍 Pose**")  # Pose title
        # st.write(pose)  # Display True/False
        pose_placeholder = st.empty()
        pose_placeholder.markdown(
            '<div class="pose-box">Waiting for pose...</div>',
            unsafe_allow_html=True
        )

    # Camera start button with popup
    if "camera_started" not in st.session_state:
        st.session_state.camera_started = False
    if "show_popup" not in st.session_state:
        st.session_state.show_popup = False

    if not st.session_state.camera_started:
        if st.button("📷 Start Camera"):
            show_instructions_popup(exercise)
            st.rerun()

    if st.session_state.show_popup:
        st.subheader("📌 Instructions")
        for step in get_exercise_instructions(exercise):
            st.write(f"✅ {step}")
        if st.button("OK"):
            st.session_state.show_popup = True
            st.session_state.camera_started = True
            st.rerun()

    # Run camera feed
    if st.session_state.camera_started:
        run_camera_feed(pose_placeholder)



# Function to run Camera Feed
def run_camera_feed(pose_placeholder):
    global pose_sequences 
    global collecting_real_time 
    frame_placeholder = st.empty()
    stop_button = st.button("⏹ Stop Camera")

    cap = cv2.VideoCapture(0)  # Access webcam

    if not cap.isOpened():
        st.error("⚠️ Error: Could not access webcam.")
        return

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Detect pose
            results = pose.process(image)
            image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Draw pose landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2))

                # Define angles to calculate

                angles_to_calculate = get_angles_to_calculate(landmarks, mp_pose)
                
                # Compute angles
                angles = [calculate_angle(*angles_to_calculate[key]) for key in angles_to_calculate]
                print(f"Calculated Angles: {angles}")

                # Input is normalized and added to sequence
                normalized_angles = scaler.transform([angles])
                pose_sequences.append(normalized_angles[0])
                
                # Collecting real-time frames(it checks if 20 new angles are appended,later forms paird of predictions and their actual angles)
                if collecting_real_time:#if flag is true
                    real_time_storage.append(angles)
                    frame_count += 1  # Count frames collected
                    if frame_count == OUTPUT_WINDOW:  # Ensure it's the same number as predicted
                        # Store the actual vs predicted values correctly
                        predicted_vs_real_storage.append((predicted_angles.copy(), real_time_storage.copy())) 

                        # Extract key angles for evaluation
                        predicted_focus = [[list(pred[3:5]) for pred in group[0]] for group in predicted_vs_real_storage]  
                        real_focus = [[list(real[3:5])for real in group[1]] for group in predicted_vs_real_storage]  

                        focus_angles = list(zip(predicted_focus, real_focus))

                        # Reset collection
                        collecting_real_time = False  
                        real_time_storage = []  
                        frame_count = 0 
                        
                # Prediction
                if len(pose_sequences) == INPUT_WINDOW  and not collecting_real_time:# if flag is false
                    INPUT_WINDOW_degrees = scaler.inverse_transform(np.array(pose_sequences))
                    print(f" Input Window (Degrees) Before Prediction:\n{INPUT_WINDOW_degrees}") 
                    input_seq = torch.tensor([pose_sequences], dtype=torch.float32).to(device)
                    with torch.no_grad():
                        predicted_normalized = lstm_model(input_seq).cpu().numpy().squeeze(0)
                    predicted_angles = scaler.inverse_transform(predicted_normalized)
                    
                    print(f"Predicted Angles:{predicted_angles}")
                    real_time_storage = []  
                    collecting_real_time = True  
                    frame_count = 0 
                    
                    # Sliding window for input
                    pose_sequences = deque(list(pose_sequences)[PRED_FREQ:], maxlen=INPUT_WINDOW)#removes previous 20 frames
                    pose_sequences.extend(real_time_storage[:PRED_FREQ])#appends new 20 frames to remaining 30 frames
                    real_time_storage = []  
                    collecting_real_time = True  
                    
                    # i have not focused on camera written display 
                    for i, angle in enumerate(predicted_angles[0]):
                        cv2.putText(image, f'Predicted {i+1}: {int(angle)}', (50, 300 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


            # Display frame in Streamlit
            frame_placeholder.image(image, channels="RGB",width=1080)
            get_pose(pose_placeholder)

            if stop_button:
                st.session_state.camera_started = False
                cap.release()
                st.rerun()
                break

    cap.release()

# Main Function to Control Page Rendering
def main():
    if "exercise" not in st.session_state:
        st.session_state.exercise = "squat"

    if "page" not in st.session_state or st.session_state.page == "home":
        display_exercise_page(st.session_state.exercise)
    elif st.session_state.page == "graphs":
        # display_graphs()
        pass
    elif st.session_state.page == "chatbot":
        # display_chatbot()
        pass

# Run the App
if __name__ == "__main__":
    main()