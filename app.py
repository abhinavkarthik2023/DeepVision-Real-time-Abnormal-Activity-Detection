import os
import cv2
import json
import numpy as np
import streamlit as st
import tempfile
from tensorflow.keras.models import load_model
import imutils

@st.cache(allow_output_mutation=True)
def load_our_model():
    return load_model(os.path.join("model", "saved_model.keras"))
    
def mean_squared_loss(x1, x2):
    diff = x1 - x2
    a, b, c, d, e = diff.shape
    n_samples = a * b * c * d * e
    sq_diff = diff ** 2
    total = sq_diff.sum()
    distance = np.sqrt(total)
    mean_distance = distance / n_samples

    return mean_distance

# Function to perform anomaly detection on video frames and save abnormal frames
def detect_anomalies(video_file_path, model):
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return

    frame_count = 0
    im_frames = []
    abnormal_frames = []
    os.makedirs('abnormality_app', exist_ok=True)
   
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        image = imutils.resize(frame, width=700)
        frame = cv2.resize(frame, (227, 227), interpolation=cv2.INTER_AREA)
        gray = 0.2989 * frame[:,:,0] + 0.5870 * frame[:,:,1] + 0.1140 * frame[:,:,2]
        gray = (gray - gray.mean()) / gray.std()
        gray = np.clip(gray, 0, 1)
        im_frames.append(gray)

        if frame_count % 10 == 0:
            im_frames = np.array(im_frames).reshape(1, 227, 227, 10, 1)
            output = model.predict(im_frames)
            loss = mean_squared_loss(im_frames, output)

            if 0.00032 < loss < 0.00038:
                st.error('🚨 Abnormal Event Detected 🚨')
                st.image(image, channels="BGR")
                cv2.imwrite(os.path.join('abnormality_app', f'frame_{frame_count}.jpg'), image)
                abnormal_frames.append(frame_count)
            im_frames = []
            
    cap.release()
   

    # Save abnormal frames indices to JSON
    with open('abnormal_frames.json', 'w') as json_file:
        json.dump(abnormal_frames, json_file)

# Streamlit code for file upload and anomaly detection
model = load_our_model()
st.title("DeepVision: Real-time Abnormal Activity Detection")
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        detect_anomalies(tmp_file.name, model)
        os.unlink(tmp_file.name)
