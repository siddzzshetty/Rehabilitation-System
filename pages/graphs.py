import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from st_helper import display_sidebar

st.set_page_config(page_title="Fitness Exercise Assistant", layout="wide")
display_sidebar()

st.title("📈 Performance Graphs")

# Ensure data file exists
data_file = "session_accuracy.csv"
if not os.path.exists(data_file):
    df = pd.DataFrame(columns=["Timestamp", "Accuracy"])
    df.to_csv(data_file, index=False)

def show_real_time_graph():
    try:
        # Read CSV
        df = pd.read_csv("session_accuracy.csv", usecols=["Timestamp", "Accuracy"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
        df["Date"] = df["Timestamp"].dt.strftime('%Y-%m-%d')

        
        # Identify sessions based on time gap (Assuming > 5 minutes is a new session)
        df['Time_Diff'] = df['Timestamp'].diff().dt.total_seconds().fillna(0)
        df['Session'] = (df['Time_Diff'] > 300).cumsum()

        # Calculate mean accuracy for each session
        session_summary = df.groupby(["Date", "Session"])["Accuracy"].mean().reset_index()

        # Plot graph
        fig, ax = plt.subplots()
        ax.scatter(session_summary["Date"], session_summary["Accuracy"], color="blue", s=100)
        ax.set_xlabel("Session Date (YYYY-MM-DD)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Session-Wise Accuracy")
        ax.set_ylim(0, 100)
        ax.grid(True)

        for i, row in session_summary.iterrows():
            ax.annotate(f"{row['Accuracy']:.2f}%", (row["Date"], row["Accuracy"]),
                        textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color='black')

        st.pyplot(fig)

    except Exception as e:
        st.write(f"Error displaying graph: {e}")

show_real_time_graph()
