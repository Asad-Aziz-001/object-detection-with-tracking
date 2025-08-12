# objectDetection.py
import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from sort import KalmanBoxTracker, iou as bb_iou

# ------------------------------
# PAGE CONFIG & CSS
# ------------------------------
st.set_page_config(page_title="AI Object Detection", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        background: linear-gradient(to right, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px 0;
    }
    .stButton button {
        background-color: #0072ff;
        color: white;
        border-radius: 10px;
        padding: 8px 20px;
        border: none;
    }
    .stButton button:hover {
        background-color: #005bd1;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🚀 Modern Object Detection & Tracking</div>', unsafe_allow_html=True)
st.write("### Real-time detection using YOLOv8 + SORT tracker")

# ------------------------------
# SIDEBAR - CONTROL PANEL
# ------------------------------
st.sidebar.header("⚙️ Control Panel")
source_type = st.sidebar.radio("Select Input Source:", ["Webcam", "Upload Video", "Upload Image"])
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
iou_match_thresh = st.sidebar.slider("IOU Match Threshold", 0.0, 1.0, 0.3, 0.05)

# ------------------------------
# LOAD MODEL
# ------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ------------------------------
# IOU WRAPPER
# ------------------------------
def iou(bb1, bb2):
    bb1 = np.array(bb1, dtype=float)
    bb2 = np.array(bb2, dtype=float)
    return bb_iou(bb1, bb2)

# ------------------------------
# PROCESS VIDEO / WEBCAM
# ------------------------------
def process_video(video_source):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("Unable to open video source.")
        return

    if "run" not in st.session_state:
        st.session_state.run = False

    start_btn = st.button("▶️ Start")
    stop_btn = st.button("🔴 Stop")
    if start_btn:
        st.session_state.run = True
    if stop_btn:
        st.session_state.run = False

    st_frame = st.empty()
    trackers = []

    while True:
        if not st.session_state.get("run", False):
            st_frame.info("Click ▶️ Start to begin detection")
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(frame_rgb, conf=conf_thresh, verbose=False)[0]

        detections = []
        for box in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, score, cls = box
            if score < conf_thresh:
                continue
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            detections.append({'bbox': [x1, y1, x2, y2], 'score': float(score), 'cls': int(cls)})

        # Predict trackers
        for tr in trackers:
            tr.predict()

        # Match detections to trackers
        matched_det_idxs = set()
        matched_tracker_idxs = set()
        if detections and trackers:
            iou_matrix = np.zeros((len(detections), len(trackers)), dtype=float)
            for d_idx, det in enumerate(detections):
                for t_idx, tr in enumerate(trackers):
                    iou_matrix[d_idx, t_idx] = iou(det['bbox'], tr.get_state())

            while True:
                d_idx, t_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                if iou_matrix[d_idx, t_idx] < iou_match_thresh:
                    break
                trackers[t_idx].update(detections[d_idx]['bbox'])
                matched_det_idxs.add(d_idx)
                matched_tracker_idxs.add(t_idx)
                iou_matrix[d_idx, :] = -1
                iou_matrix[:, t_idx] = -1

        # Add unmatched detections as new trackers
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det_idxs:
                trackers.append(KalmanBoxTracker(det['bbox']))

        # Remove old trackers
        trackers = [t for t in trackers if t.time_since_update <= 30]

        # Draw trackers
        for tr in trackers:
            x1, y1, x2, y2 = map(int, tr.get_state())
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            label = f"ID {tr.id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{model.names[det['cls']]} {det['score']:.2f}"
            cv2.putText(frame, label, (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        st_frame.image(frame, channels="BGR", use_container_width=True)

    cap.release()

# ------------------------------
# PROCESS IMAGE
# ------------------------------
def process_image(image):
    frame = np.array(image)
    results = model.predict(frame, conf=conf_thresh, verbose=False)[0]
    for box in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, score, cls = box
        if score < conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{model.names[int(cls)]} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    st.image(frame, channels="RGB", use_container_width=True)

# ------------------------------
# INPUT HANDLING
# ------------------------------
if source_type == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.success("Video uploaded successfully!")
        if st.button("▶️ Start Video"):
            st.session_state.run = True
            process_video(tfile.name)

elif source_type == "Webcam":
    if st.button("▶️ Start Webcam"):
        st.session_state.run = True
        process_video(0)

elif source_type == "Upload Image":
    uploaded_img = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        st.image(uploaded_img, caption="Uploaded Image", use_container_width=True)
        if st.button("🔍 Detect Objects"):
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            process_image(img)

st.markdown("---")
st.markdown("""
    <div style="text-align: center;">
        <p style="color: #7f8c8d;">© 2025 Advanced Object Tracker | Developed by Asad-Aziz</p>
    </div>
""", unsafe_allow_html=True)
