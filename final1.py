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

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, bc = b - a, c - b
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

lstm_model = torch.jit.load(r'model_squat_30_10.pt')#for linux relative path
lstm_model.to(device)
lstm_model.eval()



scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(np.array([[0] * 7, [180] * 7]))

input_window= 30
output_window = 10
pred_freq = 5

pose_sequences = deque(maxlen=input_window )
real_time_storage = []
frame_count = 0
collecting_real_time = False #this is a flag.false-during prediction, true-calculating real time values. 
predicted_vs_real_storage = [] 
focus_angles=[]



# Allow Streamlit to run in Jupyter Notebook
nest_asyncio.apply()

# Set up Streamlit Page Configuration
st.set_page_config(page_title="Fitness Exercise Assistant", layout="wide")

# Set your Groq API key (Make sure to replace this with your actual API key)
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual API key

# Initialize the LangChain ChatGroq model
chat_model = ChatGroq(model_name="llama-3.3-70b-versatile")  # Ensure using a supported model

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm your AI coach. How can I help with your exercise?")
    ]

# Initialize DUMMY_VARIABLE
DUMMY_VARIABLE = True  # Placeholder variable, will be integrated with .pkl

# Sidebar Navigation
st.sidebar.title("Navigation")
if st.sidebar.button("📊 View Graphs"):
    st.session_state.page = "graphs"
    st.rerun()

if st.sidebar.button("💬 Chatbot"):
    st.session_state.page = "chatbot"
    st.rerun()

st.sidebar.subheader("Select Exercise")
exercises = {
    "🏋️ Squat": "squat",
    "🧎 Sit Up": "situp",
    "👐 Push-up": "pushup",
    "💪 Pull-up": "pullup",
    "🏃 Jumping Jacks": "jumpingjacks"
}

for label, key in exercises.items():
    if st.sidebar.button(label):
        st.session_state.exercise = key
        st.rerun()

# Function to display Graphs
def display_graphs():
    st.title("📈 Performance Graphs")
    st.write("Your exercise performance graphs will appear here.")
    st.line_chart({"Squats": [10, 12, 15, 20], "Push-ups": [5, 8, 10, 12]})

# Function to handle AI Chatbot
def display_chatbot():
    st.title("💬 AI Chatbot")
    st.subheader("Chat with your AI fitness coach!")

    # Display past messages in order
    for message in st.session_state.chat_history:
        role = "assistant" if isinstance(message, AIMessage) else "user"
        with st.chat_message(role):
            st.write(message.content)

    # User Input for Chatbot
    user_input = st.chat_input("Type your question here...")

    if user_input:
        # Append user message
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        # Display user message
        with st.chat_message("user"):
            st.write(user_input)

        # Get AI response
        response = chat_with_exercise_assistant(st.session_state.chat_history, user_input)

        # Display AI response
        with st.chat_message("assistant"):
            st.write(response)

        # Store messages in history
        st.session_state.chat_history.append(AIMessage(content=response))
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
        st.write(f"🔹 **{DUMMY_VARIABLE}**")  # Display True/False

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
            st.session_state.show_popup = False
            st.session_state.camera_started = True
            st.rerun()

    # Run camera feed
    if st.session_state.camera_started:
        run_camera_feed()

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

# Function to run Camera Feed
def run_camera_feed():
    global pose_sequences 
    global collecting_real_time 
    frame_placeholder = st.empty()
    stop_button = st.button("⏹ Stop Camera")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Access webcam

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
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Draw pose landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2))

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
                    if frame_count == output_window:  # Ensure it's the same number as predicted
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
                if len(pose_sequences) == input_window  and not collecting_real_time:# if flag is false
                    input_window_degrees = scaler.inverse_transform(np.array(pose_sequences))
                    print(f" Input Window (Degrees) Before Prediction:\n{input_window_degrees}") 
                    input_seq = torch.tensor([pose_sequences], dtype=torch.float32).to(device)
                    with torch.no_grad():
                        predicted_normalized = lstm_model(input_seq).cpu().numpy().squeeze(0)
                    predicted_angles = scaler.inverse_transform(predicted_normalized)
                    
                    print(f"Predicted Angles:{predicted_angles}")
                    real_time_storage = []  
                    collecting_real_time = True  
                    frame_count = 0 
                    
                    # Sliding window for input
                    pose_sequences = deque(list(pose_sequences)[pred_freq:], maxlen=input_window)#removes previous 20 frames
                    pose_sequences.extend(real_time_storage[:pred_freq])#appends new 20 frames to remaining 30 frames
                    real_time_storage = []  
                    collecting_real_time = True  
                    
                    # i have not focused on camera written display 
                    for i, angle in enumerate(predicted_angles[0]):
                        cv2.putText(image, f'Predicted {i+1}: {int(angle)}', (50, 300 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


            # Display frame in Streamlit
            frame_placeholder.image(image, channels="RGB", use_column_width=True)

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
        display_graphs()
    elif st.session_state.page == "chatbot":
        display_chatbot()

# Run the App
if __name__ == "__main__":
    main()