import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# MediaPipe for pose detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Your Pose Transformer Model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class PoseTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super(PoseTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, 128)
        self.pos_encoder = PositionalEncoding(128)
        encoder_layers = nn.TransformerEncoderLayer(d_model=128, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Global average pooling
        x = self.fc(x)
        return x

# Load your pre-trained model (Assuming it's already trained)
model = PoseTransformer(input_dim=99, num_classes=5)  # Modify according to the number of joints
model.load_state_dict(torch.load('pose_transformer.pth'))
model.eval()

# Function to extract joint angles from the frame
def get_joint_angles(frame):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        angles = []
        for lm in landmarks:
            angles.append([lm.x, lm.y, lm.z])  # Extract x, y, z for each joint
        angles_flat = np.array(angles).flatten()  # Flatten for model input
        return angles_flat
    return None

# Real-time pose detection and correction visualization
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect pose landmarks
        results = pose.process(frame_rgb)

        # Get joint angles and make predictions using Transformer
        angles_flat = get_joint_angles(frame)
        if angles_flat is not None:
            angles_tensor = torch.tensor(angles_flat).unsqueeze(0).float()
            outputs = model(angles_tensor)
            predicted_label = torch.argmax(outputs, dim=1).item()
            
            # Define good pose by the Transformer model's output or a threshold check
            if predicted_label == 1:  # Assuming 1 is the label for incorrect pose
                correction_needed = True
            else:
                correction_needed = False

        # Render pose landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 0, 255) if correction_needed else (0, 255, 0),
                                                         thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0, 0, 255) if correction_needed else (0, 255, 0),
                                                         thickness=2, circle_radius=2))

        # Display the video with corrections
        cv2.imshow('Pose Correction', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
