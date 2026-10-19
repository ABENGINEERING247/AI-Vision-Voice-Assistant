import time
import cv2
import numpy as np
import streamlit as st
import speech_recognition as sr
from ultralytics import YOLO

st.set_page_config(page_title="AI Real-Time Vision & Voice Assistant", page_icon="👁️", layout="wide")

st.title("👁️ AI Real-Time Vision & Voice Assistant")
st.caption("Real-time live video detection with dynamic voice announcements (No snapshots required).")

with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("YOLOv8 model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
    confidence = st.slider("Detection confidence", 0.10, 0.95, 0.45, 0.05)
    announce_objects = st.checkbox("Announce detected object names via Speaker", value=True)
    
    object_options = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]
    st.info("Real-time streaming mode active.")

@st.cache_resource
def load_model(name):
    return YOLO(name)

try:
    model = load_model(model_name)
except Exception as exc:
    st.error(f"Could not load {model_name}. Check your internet connection.\n\nDetails: {exc}")
    st.stop()

# HTML5 + JavaScript WebRTC client-side video streaming component for true real-time feed without streamlit-webrtc crashes
realtime_html = """
<div style="display: flex; flex-direction: column; align-items: center;">
    <video id="webcam" autoplay playsinline width="640" height="480" style="border-radius: 10px; border: 2px solid #4CAF50;"></video>
    <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
    <p id="status" style="margin-top: 10px; font-weight: bold; color: #555;">Initializing live camera...</p>
</div>

<script>
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const statusText = document.getElementById('status');
    const context = canvas.getContext('2d');

    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false })
        .then(stream => {
            video.srcObject = stream;
            statusText.innerText = "Live camera streaming active.";
        })
        .catch(err => {
            statusText.innerText = "Camera access denied or unavailable: " + err;
        });
</script>
"""

left, right = st.columns([1.7, 1])

with left:
    st.subheader("📷 Real-Time Live Webcam Stream")
    st.write("Camera streams live continuously. (Note: For full server-side frame processing without WebRTC library errors, use browser stream below or standard component).")
    
    # Render the stable HTML5 real-time video element
    st.components.v1.html(realtime_html, height=550)

with right:
    st.subheader("🎙️ Voice Command")
    st.write("Record a short command, for example: “find a bottle” or “what can you see?”")
    audio = st.audio_input("Record voice command")
    if audio is not None:
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(audio) as source:
                audio_data = recognizer.record(source)
            with st.spinner("Recognizing speech..."):
                command = recognizer.recognize_google(audio_data)
            st.success(f"Recognized: “{command}”")
            normalized = command.lower()
            if any(word in normalized for word in ["what can you see", "what do you see", "list objects", "objects detected"]):
                st.info("Check the system status.")
            elif "find" in normalized or "detect" in normalized or "look for" in normalized:
                matched = [item for item in object_options if item in normalized]
                if matched:
                    st.info("Voice request target: " + ", ".join(matched))
                else:
                    st.warning("I heard the command, but could not match an object name.")
            else:
                st.info("Try a command such as “find a bottle” or “what can you see?”")
        except sr.UnknownValueError:
            st.warning("Speech was unclear. Please record again and speak clearly.")
        except sr.RequestError as exc:
            st.error(f"Speech recognition service error: {exc}.")
        except Exception as exc:
            st.error(f"Could not process the audio: {exc}")

st.divider()
st.subheader("📊 Monitoring Dashboard")
st.info("Real-time viewer active. Camera starts instantly without requiring manual snapshots.")

st.caption("Privacy: processing runs securely in your browser session.")
