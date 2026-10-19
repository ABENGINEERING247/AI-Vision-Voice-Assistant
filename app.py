import time
import threading
from collections import Counter, deque

import av
import cv2
import streamlit as st
import speech_recognition as sr
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

st.set_page_config(page_title="AI Vision & Voice Assistant", page_icon="👁️", layout="wide")

st.title("👁️ AI Vision & Voice Assistant")
st.caption("Live webcam object detection with spoken labels and microphone voice commands.")

with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("YOLO model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
    confidence = st.slider("Detection confidence", 0.10, 0.95, 0.45, 0.05)
    announce = st.checkbox("Announce detected objects", value=True)
    announce_interval = st.slider("Repeat announcement after (seconds)", 2, 20, 5)
    only_selected = st.checkbox("Announce selected objects only", value=False)
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
    selected_objects = st.multiselect("Objects to announce", object_options, default=["person", "bottle", "cell phone"])
    st.info("The nano model is the lightest option for cloud performance.")

@st.cache_resource
def load_model(name):
    return YOLO(name)

try:
    model = load_model(model_name)
except Exception as exc:
    st.error(f"Could not load {model_name}. Check your internet connection and installed Ultralytics version.\n\nDetails: {exc}")
    st.stop()

class VisionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.confidence = confidence
        self.announce_enabled = announce
        self.interval = announce_interval
        self.only_selected = only_selected
        self.selected = set(selected_objects)
        self.lock = threading.Lock()
        self.counts = Counter()
        self.events = deque(maxlen=30)
        self.last_spoken = {}
        self.latest_labels = []
        self.last_frame_time = time.time()
        self.fps = 0.0

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        now = time.time()
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
            self.counts.update(labels)
            self.fps = 1.0 / max(now - self.last_frame_time, 0.001)
            self.last_frame_time = now
            if self.announce_enabled:
                for label in sorted(set(labels)):
                    if self.only_selected and label not in self.selected:
                        continue
                    if now - self.last_spoken.get(label, 0) >= self.interval:
                        self.events.appendleft((time.strftime("%H:%M:%S"), label))
                        self.last_spoken[label] = now

        return av.VideoFrame.from_ndarray(image, format="bgr24")

def get_processor(ctx):
    return ctx.video_processor

left, right = st.columns([1.7, 1])
with left:
    st.subheader("📷 Live webcam")
    st.write("Allow camera access when your browser asks. For best performance, start with the nano model.")
    ctx = webrtc_streamer(
        key="ai-vision-voice-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VisionProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15}
            }, 
            "audio": False
        },
        async_processing=True,
    )
with right:
    st.subheader("🎙️ Voice command")
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
                st.info("When the webcam is running, see the latest detected labels in the monitoring panel below.")
            elif "find" in normalized or "detect" in normalized or "look for" in normalized:
                matched = [item for item in object_options if item in normalized]
                if matched:
                    st.info("Voice request target: " + ", ".join(matched))
                    st.session_state["voice_targets"] = matched
                else:
                    st.warning("I heard the command, but could not match an object name. Try “find a bottle” or “detect a person”.")
            elif "stop" in normalized:
                st.info("Use the Stop button on the webcam panel to stop the camera.")
            else:
                st.info("Try a command such as “find a bottle”, “detect a person”, or “what can you see?”")
        except sr.UnknownValueError:
            st.warning("Speech was unclear. Please record again and speak clearly.")
        except sr.RequestError as exc:
            st.error(f"Speech recognition service error: {exc}. Google recognition requires internet access.")
        except Exception as exc:
            st.error(f"Could not process the audio: {exc}")

st.divider()
st.subheader("📊 Monitoring dashboard")
if ctx and ctx.state.playing and ctx.video_processor:
    processor = ctx.video_processor
    with processor.lock:
        current_labels = list(processor.latest_labels)
        counts = dict(processor.counts)
        events = list(processor.events)
        fps = processor.fps
    c1, c2, c3 = st.columns(3)
    c1.metric("Objects in latest frame", len(current_labels))
    c2.metric("Unique labels in latest frame", len(set(current_labels)))
    c3.metric("Approx. processing FPS", f"{fps:.1f}")
    st.write("**Currently visible:** " + (", ".join(sorted(set(current_labels))) if current_labels else "No objects detected yet"))
    if counts:
        st.write("**Cumulative detections:**")
        st.dataframe([{"Object": name, "Frame detections": count} for name, count in counts.items()],
                     use_container_width=True, hide_index=True)
    st.write("**Recent announcement events:**")
    if events:
        st.dataframe([{"Time": t, "Object": label} for t, label in events],
                     use_container_width=True, hide_index=True)
    else:
        st.caption("No announcement events yet.")
else:
    st.info("Start the webcam above to see live detections and monitoring statistics.")

st.subheader("🔊 Spoken object names")
st.warning(
    "Browser-based Streamlit apps cannot reliably play audio from the server's Python process through the user's speakers. "
    "This starter app logs announcement events; the optional local TTS module below can speak on the computer running Streamlit. "
    "For a deployed cloud app, use browser-side speech synthesis for audible announcements."
)
st.markdown(
    "To enable local computer speech, install `pyttsx3` and add a local speech worker that consumes the processor's event queue. "
    "The microphone recognizer uses Google's online speech recognition service, so audio is sent to that service for transcription."
)
st.caption("Privacy: video processing runs in the app session. Do not use this demo for covert monitoring; obtain consent before recording or monitoring people.")
