# objectDetection.py
import os
import streamlit as st
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO
import warnings
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Set environment variables for headless operation
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

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
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎯 AI Object Detection & Tracking</div>', unsafe_allow_html=True)
st.write("### Real-time detection using YOLOv8 + SORT tracker")

# ------------------------------
# SIDEBAR - CONTROL PANEL
# ------------------------------
st.sidebar.header("⚙️ Control Panel")
source_type = st.sidebar.radio("Select Input Source:", ["Upload Video", "Upload Image", "Webcam"])
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# ------------------------------
# LOAD MODEL
# ------------------------------
@st.cache_resource
def load_model():
    try:
        # Download YOLOv8n model if not present
        model = YOLO("yolov8n.pt")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# ------------------------------
# SORT TRACKER IMPLEMENTATION
# ------------------------------
class KalmanBoxTracker:
    count = 0
    
    def __init__(self, bbox):
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        # Store bbox as [x1, y1, x2, y2]
        self.bbox = bbox
        self.time_since_update = 0
        
    def predict(self):
        self.time_since_update += 1
        
    def update(self, bbox):
        self.bbox = bbox
        self.time_since_update = 0
        
    def get_state(self):
        return self.bbox

def iou(bb1, bb2):
    """Calculate Intersection over Union between two bounding boxes"""
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

# ------------------------------
# PROCESS VIDEO
# ------------------------------
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Unable to open video source.")
        return
    
    # Initialize session state for tracking
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
    trackers = []
    
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
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'score': float(score),
                        'cls': int(cls)
                    })
        
        # Update trackers
        matched_det_idxs = set()
        matched_tracker_idxs = set()
        
        if detections and trackers:
            iou_matrix = np.zeros((len(detections), len(trackers)))
            for d_idx, det in enumerate(detections):
                for t_idx, tr in enumerate(trackers):
                    iou_matrix[d_idx, t_idx] = iou(det['bbox'], tr.get_state())
            
            # Greedy matching
            while True:
                max_iou = np.max(iou_matrix)
                if max_iou < 0.3:  # IOU threshold
                    break
                d_idx, t_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                trackers[t_idx].update(detections[d_idx]['bbox'])
                matched_det_idxs.add(d_idx)
                matched_tracker_idxs.add(t_idx)
                iou_matrix[d_idx, :] = -1
                iou_matrix[:, t_idx] = -1
        
        # Add new trackers
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det_idxs:
                trackers.append(KalmanBoxTracker(det['bbox']))
        
        # Update predictions
        for tr in trackers:
            tr.predict()
        
        # Remove old trackers
        trackers = [t for t in trackers if t.time_since_update <= 30]
        
        # Draw results
        frame_with_boxes = frame.copy()
        
        # Draw tracked objects
        for tr in trackers:
            x1, y1, x2, y2 = tr.get_state()
            if x2 - x1 > 0 and y2 - y1 > 0:
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID {tr.id}"
                cv2.putText(frame_with_boxes, label, (x1, max(0, y1 - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{model.names[det['cls']]} {det['score']:.2f}"
            cv2.putText(frame_with_boxes, label, (x1, y2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Display frame
        frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    
    cap.release()
    st.session_state.run_tracking = False
    st.success("✅ Video processing completed!")

# ------------------------------
# PROCESS IMAGE
# ------------------------------
def process_image(image_bytes):
    # Read image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        st.error("Failed to load image")
        return
    
    # Run YOLO detection
    results = model(img, conf=conf_thresh, verbose=False)[0]
    
    # Draw detections
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
    
    # Display results
    img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, channels="RGB", use_container_width=True)
    
    if detections:
        st.success(f"✅ Detected {len(detections)} objects")
        st.write("**Detected objects:**")
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
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        
        st.success("✅ Video uploaded successfully!")
        st.video(uploaded_file)
        
        # Process video
        process_video(tfile.name)
        
        # Cleanup
        try:
            os.unlink(tfile.name)
        except:
            pass

elif source_type == "Upload Image":
    uploaded_img = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        st.image(uploaded_img, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Detect Objects", type="primary"):
            with st.spinner("Detecting objects..."):
                process_image(uploaded_img.read())

elif source_type == "Webcam":
    st.warning("⚠️ Webcam feature requires local deployment. Streamlit Cloud does not support webcam access.")
    st.info("💡 To use webcam, run this app locally: `streamlit run objectDetection.py`")
    
    # Add instructions for local deployment
    with st.expander("📖 How to run locally"):
        st.code("""
        # Install dependencies
        pip install streamlit opencv-python ultralytics numpy
        
        # Run the app
        streamlit run objectDetection.py
        """, language="bash")

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #7f8c8d;">
        <p>Powered by <b>YOLOv8</b> + <b>SORT Tracker</b> | Developed by <b>ASAD AZIZ</b></p>
        <p style="font-size: 12px;">© 2025 Advanced Object Detection & Tracking System</p>
    </div>
""", unsafe_allow_html=True)
