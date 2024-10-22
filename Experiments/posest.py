import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import mediapipe as mp

# Step 1: Load the Dataset
angles_df = pd.read_csv('angles.csv')
labels_df = pd.read_csv('labels.csv')
landmarks_df = pd.read_csv('landmarks.csv')
xyz_distances_df = pd.read_csv('xyz_distances.csv')
calculated_distances_df = pd.read_csv('calculated_3d_distances.csv')

# Preprocess the dataset
def create_sequences(angles_df, landmarks_df, labels_df, seq_length=50):
    sequences = []
    labels = []

    for vid_id in angles_df['vid_id'].unique():
        angles_seq = angles_df[angles_df['vid_id'] == vid_id].iloc[:, 1:].values  # Exclude vid_id
        landmarks_seq = landmarks_df[landmarks_df['vid_id'] == vid_id].iloc[:, 1:].values  # Exclude vid_id
        label = labels_df[labels_df['vid_id'] == vid_id]['class'].values[0]

        # Generate sequences
        for start in range(len(angles_seq) - seq_length):
            end = start + seq_length
            angles_subseq = angles_seq[start:end]
            landmarks_subseq = landmarks_seq[start:end]

            # Combine angles and landmarks
            combined = np.hstack((angles_subseq, landmarks_subseq))
            sequences.append(combined)
            labels.append(label)

    return np.array(sequences), np.array(labels)

# Create sequences for training
sequences, labels = create_sequences(angles_df, landmarks_df, labels_df)

# Step 2: Preprocess the Labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)  # Convert to numeric labels
one_hot_labels = to_categorical(encoded_labels)

# Step 3: Split the Data
X_train, X_val, y_train, y_val = train_test_split(sequences, one_hot_labels, test_size=0.2, random_state=42)

# Step 4: Define Dataset Class
class PoseDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Create DataLoader
train_dataset = PoseDataset(X_train, y_train)
val_dataset = PoseDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Step 5: Define the Spatial-Temporal Transformer Model
class SpatialTemporalTransformer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SpatialTemporalTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=input_dim, nhead=4, num_encoder_layers=3)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x shape: (sequence_length, batch_size, input_dim)
        x = x.permute(1, 0, 2)  # Change to (batch_size, sequence_length, input_dim)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Take the mean over the sequence
        x = self.fc(x)
        return x

# Step 6: Train the Model
def train_model(model, train_loader, val_loader, num_epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += loss_fn(outputs, labels).item()
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
            
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Accuracy: {correct / total:.4f}')

# Instantiate the model and train
model = SpatialTemporalTransformer(input_dim=X_train.shape[2], num_classes=len(label_encoder.classes_))
train_model(model, train_loader, val_loader, num_epochs=10)

# Step 7: Real-Time Pose Correction Feedback
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)  # Use the default camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False

    # Get pose landmarks
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Draw landmarks on the frame
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Prepare input for the model
        landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark]).flatten()
        # Assuming the length of landmarks is 99 (3 coordinates for each of the 33 landmarks)
        input_data = np.concatenate([landmarks]).reshape(1, 50, 99)  # Assuming we need a sequence of 50 frames
        input_tensor = torch.tensor(input_data, dtype=torch.float32)

        # Get model predictions
        model.eval()
        with torch.no_grad():
            prediction = model(input_tensor)
            predicted_class = torch.argmax(prediction, dim=1).item()
            exercise_name = label_encoder.inverse_transform([predicted_class])[0]

        # Display the predicted exercise
        cv2.putText(frame, f'Predicted Exercise: {exercise_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Pose Correction Feedback', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
