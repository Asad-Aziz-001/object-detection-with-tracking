import streamlit as st
import os
import warnings
import sys
import logging

# ============================================
# SUPPRESS ALL WARNINGS
# ============================================
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=SyntaxWarning, module='filterpy')

os.environ["YOLO_CONFIG_DIR"] = "/tmp/ultralytics"
os.environ["YOLO_VERBOSE"] = "False"
os.environ["ULTRALYTICS_VERBOSE"] = "False"

logging.getLogger("ultralytics").setLevel(logging.ERROR)

try:
    import filterpy
    filterpy.common.helpers.__warningregistry__ = {}
except:
    pass

import tempfile
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="NeuralEye — Object Detection",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# GLOBAL CSS
# ============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background-color: #050709 !important;
    font-family: 'Syne', sans-serif;
    color: #e0e8f0;
}

/* Noise grain overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1400px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090c12 0%, #060810 100%) !important;
    border-right: 1px solid rgba(0,200,255,0.1) !important;
    padding-top: 0 !important;
}

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* Sidebar brand */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 1.2rem 1.4rem 1.4rem;
    border-bottom: 1px solid rgba(0,200,255,0.12);
    margin-bottom: 1.4rem;
}
.sidebar-brand-icon {
    width: 34px; height: 34px;
    background: conic-gradient(from 180deg, #00c8ff, #0055ff, #00c8ff);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    box-shadow: 0 0 18px rgba(0,200,255,0.35);
    animation: rotateBorder 6s linear infinite;
}
@keyframes rotateBorder {
    from { filter: hue-rotate(0deg); }
    to   { filter: hue-rotate(360deg); }
}
.sidebar-brand-text {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.1rem;
    letter-spacing: 0.04em;
    background: linear-gradient(90deg, #00c8ff, #7b8cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sidebar-brand-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #4a607a;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 1px;
}

/* Sidebar section labels */
.sidebar-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #00c8ff;
    padding: 0 1.4rem 0.5rem;
    margin-bottom: 0.4rem;
    margin-top: 1.2rem;
    opacity: 0.75;
}

/* Streamlit radio in sidebar */
section[data-testid="stSidebar"] .stRadio label {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    margin: 3px 0;
    border-radius: 8px;
    cursor: pointer;
    font-family: 'Syne', sans-serif;
    font-size: 0.88rem;
    font-weight: 600;
    color: #7090a8;
    transition: all 0.2s ease;
    border: 1px solid transparent;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(0,200,255,0.06);
    color: #c0d8e8;
    border-color: rgba(0,200,255,0.15);
}

/* Slider styling */
section[data-testid="stSidebar"] .stSlider > div > div > div {
    background: rgba(0,200,255,0.15) !important;
}
section[data-testid="stSidebar"] .stSlider > div > div > div > div {
    background: linear-gradient(90deg, #00c8ff, #0060ff) !important;
    box-shadow: 0 0 8px rgba(0,200,255,0.5) !important;
}

/* Slider label */
section[data-testid="stSidebar"] .stSlider label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #5880a0 !important;
    letter-spacing: 0.05em;
}

/* Slider value */
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] {
    color: #3a5570 !important;
}

/* File uploader in sidebar */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: rgba(0,200,255,0.03) !important;
    border: 1px dashed rgba(0,200,255,0.2) !important;
    border-radius: 10px !important;
    padding: 0.6rem !important;
    transition: border-color 0.3s;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
    border-color: rgba(0,200,255,0.45) !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #4a6a80 !important;
}

/* ── Hero Header ── */
.hero-wrap {
    position: relative;
    padding: 2.8rem 0 2.2rem;
    margin-bottom: 2rem;
    overflow: hidden;
}
.hero-bg-grid {
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,200,255,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,200,255,0.035) 1px, transparent 1px);
    background-size: 48px 48px;
    mask-image: radial-gradient(ellipse 90% 100% at 50% 0%, black 40%, transparent 100%);
    pointer-events: none;
}
.hero-glow-left {
    position: absolute;
    left: -80px; top: -40px;
    width: 400px; height: 280px;
    background: radial-gradient(ellipse, rgba(0,200,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-glow-right {
    position: absolute;
    right: -40px; top: 10px;
    width: 300px; height: 200px;
    background: radial-gradient(ellipse, rgba(80,80,255,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,200,255,0.07);
    border: 1px solid rgba(0,200,255,0.2);
    border-radius: 100px;
    padding: 4px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #00c8ff;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.hero-tag-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00c8ff;
    box-shadow: 0 0 6px #00c8ff;
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 5vw, 3.6rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.02em;
    color: #f0f8ff;
    margin-bottom: 0.8rem;
}
.hero-title span {
    background: linear-gradient(120deg, #00c8ff 0%, #4488ff 50%, #8866ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #4a6a88;
    letter-spacing: 0.04em;
    max-width: 560px;
}

/* ── Stats Bar ── */
.stats-bar {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: rgba(0,200,255,0.08);
    border: 1px solid rgba(0,200,255,0.1);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 2rem;
}
.stat-cell {
    background: rgba(5,7,12,0.9);
    padding: 1rem 1.4rem;
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #344a5e;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #00c8ff;
    line-height: 1;
}
.stat-value.green { color: #00e5a0; }
.stat-value.purple { color: #9988ff; }
.stat-value.orange { color: #ff9040; }

/* ── Panel / Card ── */
.panel {
    background: rgba(8,12,20,0.85);
    border: 1px solid rgba(0,200,255,0.1);
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 1.5rem;
    transition: border-color 0.3s;
}
.panel:hover {
    border-color: rgba(0,200,255,0.22);
}
.panel-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 1rem 1.4rem;
    border-bottom: 1px solid rgba(0,200,255,0.08);
    background: rgba(0,200,255,0.025);
}
.panel-header-icon {
    font-size: 1rem;
    opacity: 0.85;
}
.panel-header-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7aa0c0;
}
.panel-header-badge {
    margin-left: auto;
    background: rgba(0,200,255,0.1);
    border: 1px solid rgba(0,200,255,0.2);
    border-radius: 100px;
    padding: 2px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: #00c8ff;
    letter-spacing: 0.1em;
}
.panel-body {
    padding: 1.4rem;
}

/* ── Detection result item ── */
.det-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    border-radius: 8px;
    background: rgba(0,200,255,0.03);
    border: 1px solid rgba(0,200,255,0.07);
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    color: #88aacc;
    transition: all 0.2s;
}
.det-item:hover {
    background: rgba(0,200,255,0.06);
    border-color: rgba(0,200,255,0.18);
    color: #aaccee;
}
.det-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #00e5a0;
    box-shadow: 0 0 5px #00e5a0;
    flex-shrink: 0;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,100,200,0.25), rgba(0,180,255,0.15)) !important;
    color: #80d0ff !important;
    border: 1px solid rgba(0,200,255,0.3) !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.6rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
    box-shadow: 0 0 0 rgba(0,200,255,0) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,140,255,0.35), rgba(0,200,255,0.25)) !important;
    border-color: rgba(0,200,255,0.6) !important;
    color: #c0eeff !important;
    box-shadow: 0 0 18px rgba(0,200,255,0.2) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0060c8, #0090e0) !important;
    color: #ffffff !important;
    border-color: #00a0f0 !important;
    box-shadow: 0 4px 20px rgba(0,160,240,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0070d8, #00a8f0) !important;
    box-shadow: 0 6px 28px rgba(0,180,255,0.4) !important;
    transform: translateY(-2px) !important;
}

/* ── Alert / Info boxes ── */
.stAlert {
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
}
div[data-testid="stSuccessMessage"] {
    background: rgba(0,229,160,0.06) !important;
    border: 1px solid rgba(0,229,160,0.2) !important;
    color: #00e5a0 !important;
    border-radius: 10px !important;
}
div[data-testid="stWarningMessage"] {
    background: rgba(255,160,40,0.06) !important;
    border: 1px solid rgba(255,160,40,0.2) !important;
    color: #ffa040 !important;
    border-radius: 10px !important;
}
div[data-testid="stInfoMessage"] {
    background: rgba(0,150,255,0.06) !important;
    border: 1px solid rgba(0,150,255,0.2) !important;
    color: #60b8ff !important;
    border-radius: 10px !important;
}
div[data-testid="stErrorMessage"] {
    background: rgba(255,60,80,0.06) !important;
    border: 1px solid rgba(255,60,80,0.2) !important;
    border-radius: 10px !important;
}

/* ── Spinner ── */
.stSpinner > div > div {
    border-top-color: #00c8ff !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(0,200,255,0.08) !important;
    margin: 2rem 0 !important;
}

/* ── Image display ── */
[data-testid="stImage"] img {
    border-radius: 10px !important;
    border: 1px solid rgba(0,200,255,0.12) !important;
}

/* ── Video display ── */
video {
    border-radius: 10px !important;
    border: 1px solid rgba(0,200,255,0.12) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(0,200,255,0.04) !important;
    border: 1px solid rgba(0,200,255,0.1) !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    color: #7aa0c0 !important;
}
.streamlit-expanderContent {
    border: 1px solid rgba(0,200,255,0.08) !important;
    border-top: none !important;
    background: rgba(5,8,14,0.7) !important;
    border-radius: 0 0 10px 10px !important;
}

/* ── Columns gap ── */
[data-testid="column"] { padding: 0 0.5rem !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2rem 0 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #2a3a50;
    letter-spacing: 0.08em;
}
.footer span { color: #00c8ff; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #060810; }
::-webkit-scrollbar-thumb { background: rgba(0,200,255,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,200,255,0.4); }
</style>
""", unsafe_allow_html=True)


# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">⬡</div>
        <div>
            <div class="sidebar-brand-text">NeuralEye</div>
            <div class="sidebar-brand-sub">v2.0 · YOLOv8</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Input Source</div>', unsafe_allow_html=True)
    source_type = st.radio(
        "",
        ["📹  Upload Video", "🖼️  Upload Image", "📷  Webcam"],
        label_visibility="collapsed"
    )
    source_type = source_type.split("  ")[1]

    st.markdown('<div class="sidebar-section">Detection Config</div>', unsafe_allow_html=True)
    conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    iou_thresh  = st.slider("IOU Match Threshold",  0.0, 1.0, 0.3, 0.05)

    st.markdown('<div class="sidebar-section">Tracker Config</div>', unsafe_allow_html=True)
    max_age  = st.slider("Max Track Age (frames)", 1, 50, 30)
    min_hits = st.slider("Min Hits to Confirm",    1, 10, 3)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#2a3a50;padding:0 0.2rem;line-height:1.8;">
        MODEL &nbsp;·&nbsp; <span style="color:#00c8ff">YOLOv8n</span><br>
        TRACKER &nbsp;·&nbsp; <span style="color:#00c8ff">SORT</span><br>
        BACKEND &nbsp;·&nbsp; <span style="color:#00c8ff">OpenCV</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================
# HERO HEADER
# ============================================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-bg-grid"></div>
    <div class="hero-glow-left"></div>
    <div class="hero-glow-right"></div>
    <div class="hero-tag">
        <div class="hero-tag-dot"></div>
        Neural Vision System · Active
    </div>
    <div class="hero-title">
        Real-Time <span>Object Detection</span><br>& Tracking
    </div>
    <div class="hero-sub">
        YOLOv8 inference pipeline · SORT multi-object tracker · Live confidence scoring
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# STATS BAR
# ============================================
st.markdown("""
<div class="stats-bar">
    <div class="stat-cell">
        <div class="stat-label">Model</div>
        <div class="stat-value">YOLOv8n</div>
    </div>
    <div class="stat-cell">
        <div class="stat-label">Classes</div>
        <div class="stat-value green">80</div>
    </div>
    <div class="stat-cell">
        <div class="stat-label">Tracker</div>
        <div class="stat-value purple">SORT</div>
    </div>
    <div class="stat-cell">
        <div class="stat-label">Threshold</div>
        <div class="stat-value orange">""" + f"{conf_thresh:.2f}" + """</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    try:
        return YOLO("yolov8n.pt")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()


# ============================================
# PROCESS VIDEO
# ============================================
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Unable to open video source.")
        return

    if "run_tracking" not in st.session_state:
        st.session_state.run_tracking = False

    st.markdown("""
    <div class="panel">
        <div class="panel-header">
            <span class="panel-header-icon">🎬</span>
            <span class="panel-header-title">Live Detection Feed</span>
            <span class="panel-header-badge">STREAM</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("▶  Start", key="start_btn"):
            st.session_state.run_tracking = True
    with col2:
        if st.button("■  Stop", key="stop_btn"):
            st.session_state.run_tracking = False

    frame_placeholder = st.empty()
    tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_thresh)

    while cap.isOpened():
        if not st.session_state.get("run_tracking", False):
            frame_placeholder.info("⬡  Click ▶ Start to begin detection")
            break

        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf_thresh, verbose=False)[0]
        detections = []

        if results.boxes is not None:
            for box in results.boxes.data.cpu().numpy():
                x1, y1, x2, y2, score, cls = box
                if score >= conf_thresh:
                    detections.append([int(x1), int(y1), int(x2), int(y2), float(score), int(cls)])

        tracks = tracker.update(detections if len(detections) > 0 else [])
        frame_out = frame.copy()

        if len(tracks) > 0:
            for track in tracks:
                x1, y1, x2, y2, tid = [int(v) for v in track]
                # Neon green bounding box
                cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 230, 120), 2)
                # Track ID pill
                label = f" ID {tid} "
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame_out, (x1, max(0, y1-22)), (x1+tw+4, max(0, y1)), (0, 180, 80), -1)
                cv2.putText(frame_out, label, (x1+2, max(12, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (5, 10, 15), 1, cv2.LINE_AA)

        for det in detections:
            x1, y1, x2, y2, score, cls = det
            cls_name = model.names[cls]
            conf_txt = f"{score:.2f}"
            cv2.putText(frame_out, f"{cls_name} {conf_txt}",
                        (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 180, 255), 1, cv2.LINE_AA)

        frame_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
    st.session_state.run_tracking = False
    st.success("✓  Detection pipeline completed")


# ============================================
# PROCESS IMAGE
# ============================================
def process_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Failed to decode image.")
        return

    results = model(img, conf=conf_thresh, verbose=False)[0]
    img_out = img.copy()
    detections = []

    if results.boxes is not None:
        for box in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, score, cls = box
            if score >= conf_thresh:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls_name = model.names[int(cls)]
                label = f"{cls_name}  {score:.2f}"
                # Cyan box
                cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 210, 255), 2)
                # Label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                cv2.rectangle(img_out, (x1, max(0, y1-20)), (x1+tw+6, max(0, y1)), (0, 140, 200), -1)
                cv2.putText(img_out, label, (x1+3, max(12, y1-4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (5, 10, 20), 1, cv2.LINE_AA)
                detections.append((cls_name, score))

    col_img, col_info = st.columns([3, 1])

    with col_img:
        st.markdown("""
        <div class="panel">
            <div class="panel-header">
                <span class="panel-header-icon">🖼️</span>
                <span class="panel-header-title">Detection Output</span>
                <span class="panel-header-badge">ANNOTATED</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        img_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, channels="RGB", use_container_width=True)

    with col_info:
        st.markdown(f"""
        <div class="panel">
            <div class="panel-header">
                <span class="panel-header-icon">📊</span>
                <span class="panel-header-title">Results</span>
                <span class="panel-header-badge">{len(detections)} found</span>
            </div>
            <div class="panel-body">
        """, unsafe_allow_html=True)

        if detections:
            for cls_name, score in detections:
                pct = int(score * 100)
                bar_color = "#00e5a0" if pct >= 80 else "#00c8ff" if pct >= 60 else "#ffa040"
                st.markdown(f"""
                <div class="det-item">
                    <div class="det-dot" style="background:{bar_color};box-shadow:0 0 5px {bar_color}"></div>
                    <span>{cls_name}</span>
                    <span style="margin-left:auto;color:{bar_color}">{pct}%</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;color:#2a3a50;text-align:center;padding:1rem 0;">No objects detected</div>', unsafe_allow_html=True)

        st.markdown('</div></div>', unsafe_allow_html=True)


# ============================================
# INPUT HANDLING
# ============================================
if source_type == "Upload Video":
    uploaded_file = st.sidebar.file_uploader(
        "Drop a video file",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="visible"
    )
    if uploaded_file:
        st.success(f"✓  Video loaded: **{uploaded_file.name}**")

        st.markdown("""
        <div class="panel">
            <div class="panel-header">
                <span class="panel-header-icon">📽️</span>
                <span class="panel-header-title">Source Preview</span>
                <span class="panel-header-badge">ORIGINAL</span>
            </div>
            <div class="panel-body">
        """, unsafe_allow_html=True)
        st.video(uploaded_file)
        st.markdown('</div></div>', unsafe_allow_html=True)

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        process_video(tfile.name)

        try:
            os.unlink(tfile.name)
        except:
            pass

    else:
        st.markdown("""
        <div class="panel" style="border-style:dashed;border-color:rgba(0,200,255,0.15)">
            <div class="panel-body" style="text-align:center;padding:3.5rem 2rem;">
                <div style="font-size:2.5rem;margin-bottom:1rem;opacity:0.4">📹</div>
                <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#3a5570;margin-bottom:0.4rem;">
                    No video selected
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#233040;">
                    Upload a video file from the sidebar to begin detection
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif source_type == "Upload Image":
    uploaded_img = st.sidebar.file_uploader(
        "Drop an image file",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )
    if uploaded_img:
        col_prev, col_btn = st.columns([4, 1])
        with col_prev:
            st.markdown("""
            <div class="panel">
                <div class="panel-header">
                    <span class="panel-header-icon">🖼️</span>
                    <span class="panel-header-title">Source Image</span>
                    <span class="panel-header-badge">UPLOADED</span>
                </div>
                <div class="panel-body">
            """, unsafe_allow_html=True)
            st.image(uploaded_img, caption="", use_container_width=True)
            st.markdown('</div></div>', unsafe_allow_html=True)

        with col_btn:
            st.write("")
            st.write("")
            run = st.button("🔍  Detect Objects", type="primary")

        if run:
            with st.spinner("Running inference..."):
                process_image(uploaded_img.read())

    else:
        st.markdown("""
        <div class="panel" style="border-style:dashed;border-color:rgba(0,200,255,0.15)">
            <div class="panel-body" style="text-align:center;padding:3.5rem 2rem;">
                <div style="font-size:2.5rem;margin-bottom:1rem;opacity:0.4">🖼️</div>
                <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#3a5570;margin-bottom:0.4rem;">
                    No image selected
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#233040;">
                    Upload an image from the sidebar to run object detection
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif source_type == "Webcam":
    st.markdown("""
    <div class="panel">
        <div class="panel-header">
            <span class="panel-header-icon">📷</span>
            <span class="panel-header-title">Webcam Mode</span>
            <span class="panel-header-badge">LOCAL ONLY</span>
        </div>
        <div class="panel-body" style="display:flex;align-items:flex-start;gap:1.5rem;padding:2rem;">
            <div style="font-size:2.8rem;opacity:0.5;flex-shrink:0">📷</div>
            <div>
                <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#ffa040;margin-bottom:0.6rem;">
                    Requires local deployment
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.74rem;color:#4a6070;line-height:1.8;">
                    Streamlit Cloud does not support webcam access.<br>
                    Run locally to enable this feature:
                </div>
                <div style="margin-top:1rem;background:rgba(0,0,0,0.4);border:1px solid rgba(0,200,255,0.1);border-radius:8px;padding:0.7rem 1rem;font-family:'JetBrains Mono',monospace;font-size:0.76rem;color:#00c8ff;">
                    $ streamlit run objectDetection.py
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================
# FOOTER
# ============================================
st.markdown("""
<div class="footer">
    Powered by <span>YOLOv8</span> &nbsp;·&nbsp; <span>SORT Tracker</span> &nbsp;·&nbsp; <span>OpenCV</span>
    &nbsp;&nbsp;|&nbsp;&nbsp; Developed by <span>ASAD AZIZ</span>
</div>
""", unsafe_allow_html=True)
