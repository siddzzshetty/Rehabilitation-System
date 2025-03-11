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
from langchain_core.prompts import ChatPromptTemplate
import pickle
import time
import threading
import torch.nn as nn
from config import chat_model, exercises, GROQ_API_KEY

pose = None 
prompt_template = ChatPromptTemplate([
        ("system", """ you will be provided with angles of left knee and right knee of a patient undergoing physiotherapy. you will get angles of 3 frame runs,
         where each frame run is of 5 frames and 2 columns(left, right knee).
         you have to tell the user what is wrong in their form and how can they fix it. keep your answer concise. you have to answer in the following format.
         Incorrect: what is wrong about my current form
         Suggestion: how can i improve my current form 
        """),
        ("user", "{question}")
    ])


def save_pickle_file(pose):
    with open("data.pkl", "wb") as f:
        pickle.dump(pose, f)


def load_pickle_file():
    with open("data.pkl", "rb") as f:
        pose = pickle.load(f)
        st.write(pose)

        
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

                # print(pose)
    except Exception as e:
        print(f"Error loading pose: {e}")


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


def chatbot_ui(prompt_template):
    

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


# Function to run Camera Feed
def run_camera_feed(pose_placeholder):
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    INPUT_WINDOW= 20
    OUTPUT_WINDOW = 5
    PRED_FREQ = 5
    pose_sequences = deque(maxlen=INPUT_WINDOW)
    real_time_storage = []
    frame_count = 0
    collecting_real_time = False #this is a flag (false-during prediction, true-calculating real time values)
    predicted_vs_real_storage = [] 
    focus_angles=[]
    realtime_loss = []
    run_buffer=[]
    loss_buffer=[]
    threshold=300
    pose = None 
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    exercise  = exercises
    BUFFER_LEN = 7

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    lstm_model = torch.jit.load(r'model_path/model_squats_scripted.pt')#for linux relative path
    lstm_model.to(device)
    lstm_model.eval()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array([[0] * 7, [180] * 7])) 
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
                # print(f"Calculated Angles: {angles}")

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
                        realtime_loss.append(calculate_loss(predicted=focus_angles[-1][0], actual=focus_angles[-1][1]))

                        run_buffer.append(focus_angles[-1][1])
                        if len(run_buffer)> BUFFER_LEN:
                            run_buffer.pop(0)

                        loss_buffer.append(realtime_loss[-1])
                        if len(loss_buffer) > BUFFER_LEN:
                            loss_buffer.pop(0)

                        if loss_buffer:
                            if all(x > 500 for x in loss_buffer):
                                print("WRONG POSE******************************************************************")
                                pose_status = False
                                save_pickle_file(pose_status)
                                

                            else:
                                print("RIGHT POSE###############################################################")
                                pose_status = True
                                save_pickle_file(pose_status)


                            
                        # Reset collection
                        collecting_real_time = False  
                        real_time_storage = []  
                        frame_count = 0 
                        
                # Prediction
                if len(pose_sequences) == INPUT_WINDOW  and not collecting_real_time:# if flag is false
                    INPUT_WINDOW_degrees = scaler.inverse_transform(np.array(pose_sequences))
                    # print(f" Input Window (Degrees) Before Prediction:\n{INPUT_WINDOW_degrees}") 
                    input_seq = torch.tensor([pose_sequences], dtype=torch.float32).to(device)
                    with torch.no_grad():
                        predicted_normalized = lstm_model(input_seq).cpu().numpy().squeeze(0)
                    predicted_angles = scaler.inverse_transform(predicted_normalized)
                    
                    # print(f"Predicted Angles:{predicted_angles}")
                    real_time_storage = []  
                    collecting_real_time = True  
                    frame_count = 0 
                    
                    # Sliding window for input
                    pose_sequences = deque(list(pose_sequences)[PRED_FREQ:], maxlen=INPUT_WINDOW)#removes previous 20 frames
                    pose_sequences.extend(real_time_storage[:PRED_FREQ])#appends new 20 frames to remaining 30 frames
                    real_time_storage = []  
                    collecting_real_time = True  
                    
                    # # i have not focused on camera written display 
                    # for i, angle in enumerate(predicted_angles[0]):
                    #     cv2.putText(image, f'Predicted {i+1}: {int(angle)}', (50, 300 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


            # Display frame in Streamlit
            frame_placeholder.image(image, channels="RGB",width=1080)
            get_pose(pose_placeholder)

            if stop_button:
                st.session_state.camera_started = False
                cap.release()
                st.rerun()
                break

    cap.release()

def display_sidebar():
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    if st.sidebar.button("📊 View Graphs", key="view_graphs"):
        st.switch_page("pages/graphs.py")

    if st.sidebar.button("💬 Chatbot", key="chatbot"):
        st.switch_page("pages/chatbot_page.py")

    # if st.sidebar.button("pickle_page", key="pickle_page"):
        # st.switch_page("pages/my_pickle_page.py")

    if st.sidebar.button("Exercise", key="exercise"):
        st.switch_page("pages/final1.py")


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