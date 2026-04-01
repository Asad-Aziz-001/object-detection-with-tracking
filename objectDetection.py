# objectDetection.py
import os
import streamlit as st
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import warnings
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Set environment variable for headless operation
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# ------------------------------
# PAGE CONFIG & CSS
# ------------------------------
st.set_page_config(
    page_title="AI Object Detection & Tracking", 
    page_icon="🎯",
    layout="wide"
)

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
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #005bd1;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎯 AI Object Detection & Tracking</div>', unsafe_allow_html=True)
st.write("### Real-time detection using YOLOv8 + SORT Tracker")

# ------------------------------
# SIDEBAR - CONTROL PANEL
# ------------------------------
st.sidebar.header("⚙️ Control Panel")
source_type = st.sidebar.radio("Select Input Source:", ["Upload Video", "Upload Image", "Webcam"])
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
iou_thresh = st.sidebar.slider("IOU Match Threshold", 0.0, 1.0, 0.3, 0.05)
max_age = st.sidebar.slider("Max Track Age (frames)", 1, 50, 30, help="Maximum frames to keep a track without detection")
min_hits = st.sidebar.slider("Min Hits to Confirm Track", 1, 10, 3, help="Minimum detections to confirm a track")

# ------------------------------
# LOAD MODEL
# ------------------------------
@st.cache_resource
def load_model():
    try:
        model = YOLO("yolov8n.pt")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# ------------------------------
# PROCESS VIDEO
# ------------------------------
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Unable to open video source.")
        return
    
    if "run_tracking" not in st.session_state:
        st.session_state.run_tracking = False
    
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶️ Start Detection", key="start_btn")
        if start_btn:
            st.session_state.run_tracking = True
    with col2:
        stop_btn = st.button("🔴 Stop Detection", key="stop_btn")
        if stop_btn:
            st.session_state.run_tracking = False
    
    frame_placeholder = st.empty()
    
    # Initialize SORT tracker
    tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_thresh)
    
    while cap.isOpened():
        if not st.session_state.get("run_tracking", False):
            frame_placeholder.info("Click ▶️ Start Detection to begin")
            break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO detection
        results = model(frame, conf=conf_thresh, verbose=False)[0]
        
        detections = []
        if results.boxes is not None:
            for box in results.boxes.data.cpu().numpy():
                x1, y1, x2, y2, score, cls = box
                if score >= conf_thresh:
                    detections.append([int(x1), int(y1), int(x2), int(y2), float(score), int(cls)])
        
        # Update tracker
        if len(detections) > 0:
            tracks = tracker.update(detections)
        else:
            tracks = tracker.update([])
        
        # Draw results
        frame_with_boxes = frame.copy()
        
        # Draw tracked objects
        if len(tracks) > 0:
            for track in tracks:
                x1, y1, x2, y2, track_id = track
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID {int(track_id)}"
                cv2.putText(frame_with_boxes, label, (x1, max(0, y1 - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2, score, cls = det
            label = f"{model.names[cls]} {score:.2f}"
            cv2.putText(frame_with_boxes, label, (x1, y2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Display frame - FIXED: Changed use_container_width to width='stretch'
        frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", width='stretch')
    
    cap.release()
    st.session_state.run_tracking = False
    st.success("✅ Video processing completed!")

# ------------------------------
# PROCESS IMAGE
# ------------------------------
def process_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        st.error("Failed to load image")
        return
    
    results = model(img, conf=conf_thresh, verbose=False)[0]
    
    img_with_boxes = img.copy()
    detections = []
    
    if results.boxes is not None:
        for box in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, score, cls = box
            if score >= conf_thresh:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                label = f"{model.names[int(cls)]} {score:.2f}"
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_with_boxes, label, (x1, max(0, y1 - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                detections.append(label)
    
    # FIXED: Changed use_container_width to width='stretch'
    img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, channels="RGB", width='stretch')
    
    if detections:
        st.success(f"✅ Detected {len(detections)} objects")
        with st.expander("📋 Detection Details"):
            for det in detections:
                st.write(f"- {det}")
    else:
        st.warning("⚠️ No objects detected")

# ------------------------------
# INPUT HANDLING
# ------------------------------
if source_type == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        
        st.success("✅ Video uploaded successfully!")
        # FIXED: Changed use_container_width to width='stretch'
        st.video(uploaded_file, width='stretch')
        
        process_video(tfile.name)
        
        # Cleanup
        try:
            os.unlink(tfile.name)
        except:
            pass

elif source_type == "Upload Image":
    uploaded_img = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        # FIXED: Changed use_container_width to width='stretch'
        st.image(uploaded_img, caption="Uploaded Image", width='stretch')
        
        if st.button("🔍 Detect Objects", type="primary"):
            with st.spinner("Detecting objects..."):
                process_image(uploaded_img.read())

elif source_type == "Webcam":
    st.warning("⚠️ Webcam feature requires local deployment. Streamlit Cloud does not support webcam access.")
    st.info("💡 To use webcam, run this app locally: `streamlit run objectDetection.py`")

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #7f8c8d;">
        <p>Powered by <b>YOLOv8</b> + <b>SORT Tracker</b> | Developed by <b>ASAD AZIZ</b></p>
    </div>
""", unsafe_allow_html=True)
