# Real-Time-Object-Detection-With-Tracking

# **1. App Structure**

Title Section: A large, gradient-colored title at the top gives it a sleek, modern look.

Sidebar (Control Panel): All user controls like input source selection, confidence threshold, and IOU threshold are placed here.

Main Area: Shows either the live camera feed, uploaded video, or image with real-time detection and tracking results.

# **2. Input Sources**

You can choose:

Webcam — real-time detection from your camera.

Upload Video — process a video file frame-by-frame.

Upload Image — detect objects in a still image.

# **3. Object Detection (YOLOv8)**

The app loads a YOLOv8 model once at the start (cached so it loads fast on next runs).

Every frame (from video or webcam) is sent to YOLOv8 for prediction.

YOLOv8 returns:

Bounding box coordinates for each detected object.

Class name (e.g., person, car, dog).

Confidence score (how sure the model is).

# **4. Object Tracking (SORT Algorithm)**

The app doesn’t just detect objects — it also tracks them across frames using the SORT algorithm.

Kalman filters predict where an object will move next, and IOU (Intersection over Union) matching ensures the same object gets the same ID every frame.

This allows you to follow each object with a consistent label like “ID 3” even if it moves around.

# **5. Detection-Tracking Matching Process**

For each new frame:

All existing trackers predict where their object should be.

New detections from YOLOv8 are compared to tracker predictions using IOU.

If a detection matches a tracker above the IOU threshold, it’s assigned to that tracker.

If a detection doesn’t match any tracker, a new tracker is created.

If a tracker hasn’t been updated for too long, it’s removed.

# **6. Visual Output**

Bounding boxes are drawn around each object.

Tracker IDs are shown above the boxes.

Detected class name + confidence score is shown below the box.

Uses use_container_width=True for responsive video display that adapts to screen size.

# **7. Styling and UX**

Dark theme with custom CSS.

Rounded buttons with hover effects.

Modern sidebar sliders for thresholds.

"Start" / "Stop" buttons for video control.

Works on both desktop and mobile browser screens (responsive layout).

# **If you run it:**

Choose your source from the sidebar.

Adjust confidence and IOU thresholds for more/less strict detection.

Start detection — objects will appear with bounding boxes and IDs that follow them as they move.
