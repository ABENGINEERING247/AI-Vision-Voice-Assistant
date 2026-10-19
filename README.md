# YOLO26 Real-Time Vision & Voice Assistant

A Streamlit starter app for live webcam object detection with Ultralytics YOLO26 and microphone-based speech recognition.

## Features

- Real-time webcam stream via `streamlit-webrtc`
- YOLO26 object detection with boxes, labels, and confidence
- Detection counts and recent event log
- Configurable confidence and announcement filters
- Microphone recording and Python speech recognition using `SpeechRecognition`
- Voice command parsing for simple requests such as "find a bottle"

## Requirements

- Python 3.10–3.12 recommended
- Webcam and microphone
- Internet connection on first launch to download `yolo26n.pt`
- A browser that supports WebRTC

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`, and allow camera/microphone access.

## Model choices

The sidebar offers `yolo26n.pt`, `yolo26s.pt`, and `yolo26m.pt`. Start with `yolo26n.pt` for lower latency. The weights are downloaded automatically by Ultralytics when first used.

Official model reference: https://docs.ultralytics.com/models/yolo26/

## How speech recognition works

The app records a short clip using Streamlit's `st.audio_input`, then calls `recognize_google` from the Python `SpeechRecognition` package. This is online speech recognition and requires internet. The recognized command is parsed for object names; it is not a continuously listening microphone.

## Important audio limitation

This starter app creates announcement events in the monitoring dashboard. Python TTS libraries such as `pyttsx3` speak on the machine running Streamlit, not necessarily in the browser user's speakers. For a cloud deployment, use browser-side speech synthesis (Web Speech API) to announce labels to the visitor.

## Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `README.md` to a GitHub repository.
2. Create a Streamlit Community Cloud app from that repository.
3. Be aware that browser webcam access uses WebRTC and may require additional STUN/TURN networking support.
4. A cloud server may not provide suitable GPU acceleration. The `n` model is the most practical starting point.
5. Test camera permissions and WebRTC behavior on the deployed URL before relying on it.

## Troubleshooting

- **Model download fails:** check internet access and update Ultralytics.
- **Camera does not start:** allow camera permissions and use HTTPS or localhost.
- **No speech transcription:** allow microphone access and check internet access.
- **Low FPS:** choose `yolo26n.pt`, reduce webcam resolution, or use a GPU-enabled local environment.
- **Unsupported Python/package errors:** use Python 3.10–3.12 and recreate the virtual environment.

## Privacy

Only monitor people with appropriate notice and consent. Speech recognition through Google's service sends the recorded audio for transcription.
