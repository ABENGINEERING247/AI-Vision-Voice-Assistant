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
st.caption("Live real-time webcam object detection with YOLO26 and browser voice announcements.")

with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("YOLO26 model", ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt"], index=0)
    confidence = st.slider("Detection confidence", 0.10, 0.95, 0.45, 0.05)
    announce_person = st.checkbox("Announce 'Person' via Speaker", value=True)
    
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
    st.info("Real-time video streaming mode active. Ensure good lighting for best YOLO26 performance.")

@st.cache_resource
def load_model(name):
    return YOLO(name)

try:
    model = load_model(model_name)
except Exception as exc:
    st.error(f"Could not load {model_name}. Check your internet connection or model availability.\n\nDetails: {exc}")
    st.stop()

class RealtimeVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.confidence = confidence
        self.lock = threading.Lock()
        self.latest_labels = []
        self.person_detected_flag = False
        self.fps = 0.0
        self.last_frame_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        now = time.time()
        
        try:
            results = self.model.predict(image, conf=self.confidence, verbose=False, imgsz=640)
            result = results[0]
            labels = []
            has_person = False
            
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    label = self.model.names.get(cls_id, str(cls_id))
                    score = float(box.conf[0].item())
                    labels.append(label)
                    
                    if label == "person":
                        has_person = True
                        
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(image, (x1, y1), (x2, y2), (40, 220, 80), 2)
                    cv2.putText(image, f"{label} {score:.0%}", (x1, max(24, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 220, 80), 2)

            with self.lock:
                self.latest_labels = labels
                self.person_detected_flag = has_person
                self.fps = 1.0 / max(now - self.last_frame_time, 0.001)
                self.last_frame_time = now
        except Exception:
            # Prevents background thread from crashing and closing the stream
            pass

        return av.VideoFrame.from_ndarray(image, format="bgr24")

left, right = st.columns([1.7, 1])

with left:
    st.subheader("📷 Real-Time Live Webcam Stream")
    st.write("Allow camera access when prompted. The video stream will remain active continuously.")
    
    ctx = webrtc_streamer(
        key="yolo26-realtime-stream",
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

# Real-time Voice Announcement Trigger based on video state
person_currently_visible = False
current_fps = 0.0
current_labels_list = []

if ctx and ctx.state.playing and ctx.video_processor:
    processor = ctx.video_processor
    with processor.lock:
        person_currently_visible = processor.person_detected_flag
        current_fps = processor.fps
        current_labels_list = list(processor.latest_labels)

if announce_person and person_currently_visible:
    current_time = time.time()
    # Throttle speech announcements to avoid overlapping speech every single frame (e.g. every 5 seconds)
    if current_time - st.session_state.get("last_speech_time", 0) > 5.0:
        st.session_state["last_speech_time"] = current_time
        
        speech_script = """
        <script>
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance("Person detected!");
                utterance.rate = 1.0;
                utterance.pitch = 1.0;
                utterance.volume = 1.0;
                window.speechSynthesis.speak(utterance);
            }
        </script>
        """
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
    st.info("Start the webcam stream above to view real-time object detection statistics and live stats.")

st.caption("Privacy: video processing runs securely in your active session.")
