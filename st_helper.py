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
import pandas as pd
from config import GROQ_API_KEY, USE_MODEL, exercises, prompt_template_squat, prompt_template_pushup, llm


pose = None 

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
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))) 
    return 180 - angle


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

# Ensure data file exists
data_file = "session_accuracy.csv"
if not os.path.exists(data_file):
    df = pd.DataFrame(columns=["Timestamp", "Accuracy"])
    df.to_csv(data_file, index=False)

def add_session_accuracy(loss_buffer, THRESHOLD):
    """Logs mean accuracy for a completed session in a CSV."""
    if not loss_buffer:
        print("No data to calculate accuracy.")
        return
    
    # Calculate Accuracy
    true_count = sum(1 for x in loss_buffer if x <= THRESHOLD)
    false_count = sum(1 for x in loss_buffer if x > THRESHOLD)
    total = true_count + false_count
    accuracy = (true_count / total) * 100 if total > 0 else 0
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') # Store in DD/MM format

    # Check or create CSV
    data_file = "session_accuracy.csv"
    if not os.path.exists(data_file):
        df = pd.DataFrame(columns=["Date", "Accuracy"])
        df.to_csv(data_file,mode='a',index=False)

    # Save Session Accuracy
    df = pd.read_csv(data_file)
    new_data = pd.DataFrame({"Timestamp": [timestamp], "Accuracy": [accuracy]})
    new_data.to_csv(data_file, mode="a", header=False, index=False)

    print(f"Session Accuracy Saved: {accuracy:.2f}% on {timestamp}")

# Functions wrt llm
def save_llm_feedback(current_feedback):
    feedback_file_path = "llm_feedback.pkl"

    if os.path.exists(feedback_file_path):
        with open(feedback_file_path, "rb") as f:
            feedback_history = pickle.load(f)
    else:
        feedback_history = []
    
    feedback_history.insert(0,current_feedback)
    
    with open(feedback_file_path, "wb") as f:
        pickle.dump(feedback_history, f)

def load_llm_feedback():
    feedback_file_path = "llm_feedback.pkl"

    if os.path.exists(feedback_file_path):
        with open(feedback_file_path, "rb") as f:
            feedback_history = pickle.load(f)
        return feedback_history
    else:
        return []

def llm_feedback(llm, prompt_template, focus_angles):
    start = time.time()
    user_input = f"""
                    predicted = {focus_angles[-3][0]}, {focus_angles[-2][0]}, {focus_angles[-1][0]} \n 
                    actual = {focus_angles[-3][1]}, {focus_angles[-2][1]}, {focus_angles[-1][1]}
                    """
    format_prompt = prompt_template.format(question=user_input)
    response = llm.invoke(format_prompt).content
    save_llm_feedback(response)
    end = time.time()
    print("response time:")
    print(end-start)
    
#new fn 
def run_camera_feed(pose_placeholder):
    # Create columns with a more balanced layout
    col1, col2 = st.columns([3, 1])

    # Initialize session state variables if not existing
    if 'camera_started' not in st.session_state:
        st.session_state.camera_started = True
    if 'llm_messages' not in st.session_state:
        st.session_state.llm_messages = []
    if 'last_mtime' not in st.session_state:
        st.session_state.last_mtime = 0
    st.session_state.llm_messages = load_llm_feedback()

    # Configuration and constants
    INPUT_WINDOW = 20
    OUTPUT_WINDOW = 5
    PRED_FREQ = 5
    THRESHOLD = 300
    BUFFER_LEN = 5
    LLM_COOLDOWN = 10

    # Check if exercise is selected
    if "selected_exercise" not in st.session_state:
        st.error("No exercise selected. Please go back and select an exercise.")
        return

    # Model and device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_exercise = st.session_state.selected_exercise

    # Dynamic model path mapping
    model_paths = {
        "squat": r"model_path/model_squats_scripted_flatten.pt",
        "pushup": r"model_path/model_pushup_scripted_flatten.pt",
    }

    if selected_exercise == "squat":
        prompt_template = prompt_template_squat
    elif selected_exercise == "pushup":
        prompt_template = prompt_template_pushup
    
    # Load model
    model_path = model_paths.get(selected_exercise)
    if not model_path:
        st.error(f"No model available for exercise: {selected_exercise}")
        return

    # Prepare model and scaler
    lstm_model = torch.jit.load(model_path)
    lstm_model.to(device)
    lstm_model.eval()

    if hasattr(lstm_model, 'flatten_parameters'):
        lstm_model.flatten_parameters()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array([[0] * 7, [180] * 7]))

    # Mediapipe setup
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Placeholders and buttons
    frame_placeholder = col1.empty()
    feedback_placeholder = col2.empty()
    stop_button = st.button("⏹ Stop Camera")

    # Camera capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("⚠️ Error: Could not access webcam.")
        return

    # State tracking variables
    pose_sequences = deque(maxlen=INPUT_WINDOW)
    real_time_storage = []
    frame_count = 0
    collecting_real_time = False
    predicted_vs_real_storage = []
    focus_angles = []
    realtime_loss = []
    run_buffer = []
    loss_buffer = []
    # wrong_buffer = []
    # right_buffer = []
    cooldown_count = 0

    # Main processing loop
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as pose:
        while cap.isOpened() and st.session_state.camera_started:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Draw pose landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2))

                angles_to_calculate = get_angles_to_calculate(landmarks, mp_pose)
                angles = [calculate_angle(*angles_to_calculate[key]) for key in angles_to_calculate]

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
                        # Extract key angles for evaluation
                        if selected_exercise == "squat":
                            predicted_focus = [[list(pred[3:5]) for pred in group[0]] for group in predicted_vs_real_storage]  
                            real_focus = [[list(real[3:5])for real in group[1]] for group in predicted_vs_real_storage]  
                        elif selected_exercise == "pushup":
                            predicted_focus = [[list(pred[6:]) for pred in group[0]] for group in predicted_vs_real_storage]  
                            real_focus = [[list(real[6:])for real in group[1]] for group in predicted_vs_real_storage]   

                        focus_angles = list(zip(predicted_focus, real_focus))
                        realtime_loss.append(calculate_loss(predicted=focus_angles[-1][0], actual=focus_angles[-1][1]))

                        run_buffer.append(focus_angles[-1][1])
                        if len(run_buffer)> BUFFER_LEN:
                            run_buffer.pop(0)

                        loss_buffer.append(realtime_loss[-1])
                        if len(loss_buffer) > BUFFER_LEN:
                            loss_buffer.pop(0)

                        if USE_MODEL:
                            cooldown_count += 1
                            if all(x > THRESHOLD for x in loss_buffer):
                                # print("WRONG POSE******************************************************************")
                                pose_status = False
                                # wrong_buffer.append(run_buffer)
                                focus_len = len(focus_angles)
                                if focus_len > 20 and cooldown_count > LLM_COOLDOWN:
                                    cooldown_count = 0
                                    t1 = threading.Thread(target=llm_feedback, args=(llm, prompt_template, focus_angles))
                                    t1.start()
                                # save_pickle_file(pose_status)
                                

                            else:
                                # print("RIGHT POSE###############################################################")
                                pose_status = True
                                # right_buffer.append(run_buffer)
                                # save_pickle_file(pose_status)

                            # true_count = sum(1 for x in loss_buffer if x <= THRESHOLD)
                            # false_count = sum(1 for x in loss_buffer if x > THRESHOLD)
                            # add_session_accuracy(df, loss_buffer, THRESHOLD)
                            
                            if pose_status:
                                cv2.putText(image, f"Bool Value: {pose_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            else:
                                cv2.putText(image, f"Bool Value: {pose_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        # Reset collection
                        collecting_real_time = False  
                        real_time_storage = []  
                        frame_count = 0 
                        
                # Prediction
                if len(pose_sequences) == INPUT_WINDOW  and not collecting_real_time:# if flag is false 
                    pose_sequences_np = np.array(pose_sequences)  # Much faster
                    input_seq = torch.tensor([pose_sequences_np], dtype=torch.float32).to(device)
                    with torch.no_grad():
                        predicted_normalized = lstm_model(input_seq).cpu().numpy().squeeze(0)
                    predicted_angles = scaler.inverse_transform(predicted_normalized)
                    
                    real_time_storage = []  
                    collecting_real_time = True  
                    frame_count = 0 
                    
                    # Sliding window for input
                    pose_sequences = deque(list(pose_sequences)[PRED_FREQ:], maxlen=INPUT_WINDOW)#removes previous 20 frames
                    pose_sequences.extend(real_time_storage[:PRED_FREQ])#appends new 20 frames to remaining 30 frames
                    real_time_storage = []  
                    collecting_real_time = True  
                    


            # Display frame
            frame_placeholder.image(image, channels="RGB", width=640)
            # get_pose(pose_placeholder)
            # Update feedback (non-blocking)
            try:
                st.session_state.llm_messages = load_llm_feedback()
                pose_status_path = "llm_feedback.pkl"
                mtime = os.path.getmtime(pose_status_path)

                if mtime != st.session_state.last_mtime:
                    st.session_state.last_mtime = mtime
                    
                    if st.session_state.llm_messages:
                        message = st.session_state.llm_messages[0]
                        # print(message)
                        if "Incorrect:" in message and "Suggestion:" in message:
                            incorrect, suggestion = message.split("Suggestion:")
                            feedback_html = f"""
                                        <div style="
                                            background-color: #1e1e1e; 
                                            padding: 15px; 
                                            border-radius: 8px; 
                                            border: 1px solid #333; 
                                            color: #f5f5f5;
                                            font-size: 14px;
                                        ">
                                            <strong style='color: #ff6b6b;'>Incorrect:</strong> {incorrect.replace('Incorrect:', '').strip()}<br><br>
                                            <strong style='color: #6bff95;'>Suggestion:</strong> {suggestion.strip()}
                                        </div>
                                        """
                            feedback_placeholder.markdown(feedback_html, unsafe_allow_html=True)
                        else:
                            feedback_placeholder.markdown(message)
                    else:
                        feedback_placeholder.info("No feedback yet.")
            except FileNotFoundError:
                feedback_placeholder.warning("Feedback file not found.")

            # Stop camera if button pressed
            if stop_button:
                # with open("focus_angles.pkl", "wb") as f:
                #     pickle.dump(focus_angles, f)
                st.session_state.camera_started = False
                break

    # Cleanup
    cap.release()
    st.rerun()

# Function to run Camera Feed
# def run_camera_feed(pose_placeholder):
#     # Create columns with a more balanced layout
#     # col1, col2 = st.columns([3, 1])  # Adjusted column ratio for better balance

#     # Initialize session state variables if not existing
#     if 'camera_started' not in st.session_state:
#         st.session_state.camera_started = True
#     if 'llm_messages' not in st.session_state:
#         st.session_state.llm_messages = []
#     if 'last_mtime' not in st.session_state:
#         st.session_state.last_mtime = 0

#     # Configuration and constants
#     GROQ_API_KEY = os.getenv('GROQ_API_KEY')  # Securely fetch API key
#     INPUT_WINDOW = 20
#     OUTPUT_WINDOW = 5
#     PRED_FREQ = 5

#     # adjustable
#     THRESHOLD = 500
#     BUFFER_LEN = 3
#     LLM_COOLDOWN = 10
    
#     # Check if exercise is selected
#     if "selected_exercise" not in st.session_state:
#         st.error("No exercise selected. Please go back and select an exercise.")
#         return
    
#     # Model and device setup
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     selected_exercise = st.session_state.selected_exercise

#     # Dynamic model path mapping
#     model_paths = {
#         "squats": r"model_path/model_squats_scripted.pt",
#         "pushups": r"model_path/model_pushup_scripted.pt",
#     }

#     # Load model
#     model_path = model_paths.get(selected_exercise)
#     if not model_path:
#         st.error(f"No model available for exercise: {selected_exercise}")
#         return

#     # Prepare model and scaler
#     lstm_model = torch.jit.load(model_path)
#     lstm_model.to(device)
#     lstm_model.eval()

#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaler.fit(np.array([[0] * 7, [180] * 7]))

#     # Mediapipe setup
#     mp_pose = mp.solutions.pose
#     mp_drawing = mp.solutions.drawing_utils

#     # Placeholders and buttons
#     frame_placeholder = col1.empty()
#     feedback_placeholder = col2.empty()
#     stop_button = st.button("⏹ Stop Camera")

#     # Camera capture
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         st.error("⚠️ Error: Could not access webcam.")
#         return

#     # State tracking variables
#     pose_sequences = deque(maxlen=INPUT_WINDOW)
#     real_time_storage = []
#     frame_count = 0
#     collecting_real_time = False
#     predicted_vs_real_storage = []
#     focus_angles = []
#     realtime_loss = []
#     run_buffer = []
#     loss_buffer = []
#     wrong_buffer = []
#     right_buffer = []
#     cooldown_count = 0

#     if not cap.isOpened():
#         st.error("⚠️ Error: Could not access webcam.")
#         return

#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as pose:
#         while cap.isOpened():
#             with col1:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 llm_messages = load_llm_feedback()
#                 st.session_state.llm_messages = llm_messages
#                 # Convert frame to RGB
#                 image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 image.flags.writeable = False
                
#                 # Detect pose
#                 results = pose.process(image)
#                 image.flags.writeable = True
#                 # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
#                 if results.pose_landmarks:
#                     landmarks = results.pose_landmarks.landmark

#                     # Draw pose landmarks
#                     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                             mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
#                                             mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2))

#                     # Define angles to calculate

#                     angles_to_calculate = get_angles_to_calculate(landmarks, mp_pose)
                    
#                     # Compute angles
#                     angles = [calculate_angle(*angles_to_calculate[key]) for key in angles_to_calculate]
#                     # print(f"Calculated Angles: {angles}")

#                     # Input is normalized and added to sequence
#                     normalized_angles = scaler.transform([angles])
#                     pose_sequences.append(normalized_angles[0])
                    
#                     # Collecting real-time frames(it checks if 20 new angles are appended,later forms paird of predictions and their actual angles)
#                     if collecting_real_time:#if flag is true
#                         real_time_storage.append(angles)
#                         frame_count += 1  # Count frames collected
#                         if frame_count == OUTPUT_WINDOW:  # Ensure it's the same number as predicted
#                             # Store the actual vs predicted values correctly
#                             predicted_vs_real_storage.append((predicted_angles.copy(), real_time_storage.copy())) 

#                             # Extract key angles for evaluation
#                             predicted_focus = [[list(pred[3:5]) for pred in group[0]] for group in predicted_vs_real_storage]  
#                             real_focus = [[list(real[3:5])for real in group[1]] for group in predicted_vs_real_storage]  

#                             focus_angles = list(zip(predicted_focus, real_focus))
#                             realtime_loss.append(calculate_loss(predicted=focus_angles[-1][0], actual=focus_angles[-1][1]))

#                             run_buffer.append(focus_angles[-1][1])
#                             if len(run_buffer)> BUFFER_LEN:
#                                 run_buffer.pop(0)

#                             loss_buffer.append(realtime_loss[-1])
#                             if len(loss_buffer) > BUFFER_LEN:
#                                 loss_buffer.pop(0)

#                             if USE_MODEL:
#                                 cooldown_count += 1
#                                 # print(cooldown_count)
#                                 if all(x > THRESHOLD for x in loss_buffer):
#                                     # print("WRONG POSE******************************************************************")
#                                     pose_status = False
#                                     wrong_buffer.append(run_buffer)
#                                     focus_len = len(focus_angles)
#                                     if focus_len > 20 and cooldown_count > LLM_COOLDOWN:
#                                         cooldown_count = 0
#                                         t1 = threading.Thread(target=llm_feedback, args=(llm, prompt_template, run_buffer))
#                                         t1.start()
#                                         # llm_output = llm_feedback(llm, prompt_template, run_buffer)
#                                         # print(focus_len,llm_output)
#                                     save_pickle_file(pose_status)
                                    

#                                 else:
#                                     # print("RIGHT POSE###############################################################")
#                                     pose_status = True
#                                     right_buffer.append(run_buffer)
#                                     save_pickle_file(pose_status)

#                                 # true_count = sum(1 for x in loss_buffer if x <= THRESHOLD)
#                                 # false_count = sum(1 for x in loss_buffer if x > THRESHOLD)
#                                 # add_session_accuracy(df, loss_buffer, THRESHOLD)
                                
#                             # Reset collection
#                             collecting_real_time = False  
#                             real_time_storage = []  
#                             frame_count = 0 
                            
#                     # Prediction
#                     if len(pose_sequences) == INPUT_WINDOW  and not collecting_real_time:# if flag is false
#                         # INPUT_WINDOW_degrees = scaler.inverse_transform(np.array(pose_sequences))
#                         # print(f" Input Window (Degrees) Before Prediction:\n{INPUT_WINDOW_degrees}") 
#                         input_seq = torch.tensor([pose_sequences], dtype=torch.float32).to(device)
#                         with torch.no_grad():
#                             predicted_normalized = lstm_model(input_seq).cpu().numpy().squeeze(0)
#                         predicted_angles = scaler.inverse_transform(predicted_normalized)
                        
#                         # print(f"Predicted Angles:{predicted_angles}")
#                         real_time_storage = []  
#                         collecting_real_time = True  
#                         frame_count = 0 
                        
#                         # Sliding window for input
#                         pose_sequences = deque(list(pose_sequences)[PRED_FREQ:], maxlen=INPUT_WINDOW)#removes previous 20 frames
#                         pose_sequences.extend(real_time_storage[:PRED_FREQ])#appends new 20 frames to remaining 30 frames
#                         real_time_storage = []  
#                         collecting_real_time = True  
                        
#                         # # i have not focused on camera written display 
#                         # for i, angle in enumerate(predicted_angles[0]):
#                         #     cv2.putText(image, f'Predicted {i+1}: {int(angle)}', (50, 300 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


#                 # Display frame in Streamlit
#                 frame_placeholder.image(image, channels="RGB",width=720)
#                 get_pose(pose_placeholder)

            
#             with col2:
#                 try:
#                     pose_status_path = "llm_feedback.pkl"
#                     mtime = os.path.getmtime(pose_status_path)

#                     if mtime != st.session_state.last_mtime:
#                         # print("yayyy")
#                         st.session_state.last_mtime = mtime
                        
#                         if st.session_state.llm_messages:
#                             message = st.session_state.llm_messages[0]
#                             # print(message)
#                             if "Incorrect:" in message and "Suggestion:" in message:
#                                 incorrect, suggestion = message.split("Suggestion:")
#                                 feedback_html = f"""
#                                     <div style="
#                                         background-color: #1e1e1e; 
#                                         padding: 15px; 
#                                         border-radius: 8px; 
#                                         border: 1px solid #333; 
#                                         color: #f5f5f5;
#                                         font-size: 14px;
#                                     ">
#                                         <strong style='color: #ff6b6b;'>Incorrect:</strong> {incorrect.replace('Incorrect:', '').strip()}<br><br>
#                                         <strong style='color: #6bff95;'>Suggestion:</strong> {suggestion.strip()}
#                                     </div>
#                                     """
#                                 feedback_placeholder.markdown(feedback_html, unsafe_allow_html=True)
#                             else:
#                                 feedback_placeholder.markdown(message)
#                         else:
#                             feedback_placeholder.info("No feedback yet.")
#                 except FileNotFoundError:
#                     feedback_placeholder.warning("Feedback file not found.")


#                 if stop_button:
#                     st.session_state.camera_started = False
#                     cap.release()
#                     st.rerun()
#                     break
#     cap.release()




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

    if st.sidebar.button("Feedback", key="feedback"):
        st.switch_page("pages/chatbot_display.py")


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

