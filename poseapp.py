import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random
from datetime import datetime, timedelta

st.title("Pose Correction Application")
st.sidebar.title("Exercise Analytics")

# Function to display real-time video feed
def open_camera(exercise):
    st.write(f"Starting {exercise} exercise. Please position yourself in front of the camera.")
    camera_placeholder = st.empty()
    camera_feed = camera_placeholder.camera_input("Exercise Camera Feed")
    
    if camera_feed is not None:
        st.write("Camera is active. Perform your exercise now.")
        if st.button("Stop Exercise"):
            camera_placeholder.empty()
            st.write("Exercise stopped.")

st.header("Select an exercise to start:")
exercises = ["Push-up", "Pull-up", "Sit-up", "Jumping Jack", "Squats"]

# Create a row of buttons
cols = st.columns(len(exercises))
for idx, exercise in enumerate(exercises):
    if cols[idx].button(exercise):
        open_camera(exercise)

if st.sidebar.button("Show Exercise Progress"):
    
    # Generate random data for line graph
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    values = [random.randint(0, 100) for _ in range(len(dates))]
    df = pd.DataFrame({'Date': dates, 'Value': values})

    # Create line graph
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Date'], df['Value'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Progress')
    ax.set_title('Exercise Progress Over Time')
    ax.grid(True)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))

    st.sidebar.pyplot(fig)

if st.sidebar.button("Show Exercise Distribution"):
    # Generate random data for pie chart
    exercise_counts = {exercise: random.randint(1, 100) for exercise in exercises}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(list(exercise_counts.values()), labels=exercise_counts.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title('Distribution of Exercises')

    st.sidebar.pyplot(fig)