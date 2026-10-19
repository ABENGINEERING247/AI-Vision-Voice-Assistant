import time
import threading
from collections import Counter, deque
import av
import cv2
import streamlit as st
import speech_recognition as sr
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

st.set_page_config(page_title="AI Real-Time Vision & Voice Assistant", page_icon="👁️", layout="wide")

st.title("👁️ AI Real-Time Vision & Voice Assistant")
st.caption("Live real-time webcam video detection with dynamic object name speech announcements.")

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
    st.info("Real-time video streaming active. Detected object names will be spoken automatically.")

@st.cache_resource
def load_model(name):
    return YOLO(name)

try:
    model = load_model(model_name)
except Exception as exc:
    st.error(f"Could not load {model_name}. Check your internet connection.\n\nDetails: {exc}")
    st.stop()

class RealtimeVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.confidence = confidence
        self.lock = threading.Lock()
        self.latest_labels = []
        self.fps = 0.0
        self.last_frame_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        now = time.time()
        
        try:
            results = self.model.predict(image, conf=self.confidence, verbose=False, imgsz=640)
            result = results[0]
            labels = []
            
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    label = self.model.names.get(cls_id, str(cls_id))
                    score = float(box.conf[0].item())
                    labels.append(label)
                        
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(image, (x1, y1), (x2, y2), (40, 220, 80), 2)
                    cv2.putText(image, f"{label} {score:.0%}", (x1, max(24, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 220, 80), 2)

            with self.lock:
                self.latest_labels = labels
                self.fps = 1.0 / max(now - self.last_frame_time, 0.001)
                self.last_frame_time = now
        except Exception:
            pass

        return av.VideoFrame.from_ndarray(image, format="bgr24")

left, right = st.columns([1.7, 1])

with left:
    st.subheader("📷 Real-Time Live Webcam Stream")
    st.write("Allow browser camera access and click 'Start' to begin live video detection.")
    
    ctx = webrtc_streamer(
        key="yolov8-realtime-dynamic-audio",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=RealtimeVideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 20}
            },
            "audio": False
        },
        async_processing=True,
    )

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
                st.info("Check the live dashboard below for current detections.")
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

current_fps = 0.0
current_labels_list = []

if ctx and ctx.state.playing and ctx.video_processor:
    processor = ctx.video_processor
    with processor.lock:
        current_fps = processor.fps
        current_labels_list = list(processor.latest_labels)

if announce_objects and current_labels_list:
    target_object = current_labels_list[0]
    current_time = time.time()
    
    if "last_spoken_times" not in st.session_state:
        st.session_state["last_spoken_times"] = {}
    
    last_spoken = st.session_state["last_spoken_times"].get(target_object, 0)
    if current_time - last_spoken > 4.0:
        st.session_state["last_spoken_times"][target_object] = current_time
        
        # Safe string concatenation avoiding any f-string curly-brace syntax conflict
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
st.subheader("📊 Real-Time Monitoring Dashboard")

if ctx and ctx.state.playing:
    c1, c2, c3 = st.columns(3)
    c1.metric("Objects in Live Frame", len(current_labels_list))
    c2.metric("Unique Labels", len(set(current_labels_list)))
    c3.metric("Processing FPS", f"{current_fps:.1f}")
    
    st.write("**Currently visible:** " + (", ".join(sorted(set(current_labels_list))) if current_labels_list else "Scanning for objects..."))
else:
    st.info("Start the webcam stream above to view real-time object detection and audio announcements.")

st.caption("Privacy: real-time video processing runs securely in your active browser session.")
