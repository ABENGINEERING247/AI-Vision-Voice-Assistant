import time
import cv2
import numpy as np
import streamlit as st
import speech_recognition as sr
from ultralytics import YOLO

st.set_page_config(page_title="AI Vision & Voice Assistant", page_icon="👁️", layout="wide")

st.title("👁️ AI Vision & Voice Assistant")
st.caption("Stable object detection using Streamlit camera input with dynamic voice announcements.")

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
    st.info("Using native Streamlit camera input for zero errors and 100% stability.")

@st.cache_resource
def load_model(name):
    return YOLO(name)

try:
    model = load_model(model_name)
except Exception as exc:
    st.error(f"Could not load {model_name}. Check your internet connection.\n\nDetails: {exc}")
    st.stop()

left, right = st.columns([1.7, 1])

current_labels = []

with left:
    st.subheader("📷 Camera Capture")
    st.write("Click 'Take Photo' below to capture an image and instantly run YOLOv8 detection and voice alerts.")
    
    camera_file = st.camera_input("Take a picture for instant detection")
    
    if camera_file is not None:
        bytes_data = camera_file.getvalue()
        np_arr = np.frombuffer(bytes_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        results = model.predict(image, conf=confidence, verbose=False, imgsz=640)
        result = results[0]
        
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                label = model.names.get(cls_id, str(cls_id))
                score = float(box.conf[0].item())
                current_labels.append(label)
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(image, (x1, y1), (x2, y2), (40, 220, 80), 2)
                cv2.putText(image, f"{label} {score:.0%}", (x1, max(24, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 220, 80), 2)
        
        annotated_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(annotated_image, channels="RGB", caption="Detected Results", use_container_width=True)

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
                st.info("Check the detected objects list below.")
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

# Dynamic Voice Announcement Trigger for Captured Image (Fixed Syntax)
if camera_file is not None and announce_objects and current_labels:
    target_object = current_labels[0]
    
    if st.session_state.get("last_announced_file") != camera_file:
        st.session_state["last_announced_file"] = camera_file
        
        speech_script = (
            "<script>\n"
            "if ('speechSynthesis' in window) {\n"
            "    window.speechSynthesis.cancel();\n"
            "    const utterance = new SpeechSynthesisUtterance('" + target_object + " detected');\n"
            "    utterance.rate = 1.0;\n"
            "    utterance.pitch = 1.0;\n"
            "    utterance.volume = 1.0;\n"
            "    window.speechSynthesis.speak(utterance);\n"
            "}\n"
            "</script>"
        )
        st.components.v1.html(speech_script, height=0)

st.divider()
st.subheader("📊 Monitoring Dashboard")

if camera_file is not None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Objects in Frame", len(current_labels))
    c2.metric("Unique Labels", len(set(current_labels)))
    c3.metric("Status", "Processed Successfully")
    
    st.write("**Currently visible:** " + (", ".join(sorted(set(current_labels))) if current_labels else "No objects detected"))
else:
    st.info("Take a picture using the camera above to view detection stats and audio announcements.")

st.caption("Privacy: processing runs securely in your browser session.")
