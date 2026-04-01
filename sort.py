"""
SORT: Simple Online and Realtime Tracking
Implementation based on: https://github.com/abewley/sort
"""

import numpy as np
from filterpy.kalman import KalmanFilter

def iou(bb1, bb2):
    """
    Calculate Intersection over Union between two bounding boxes
    """
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects.
    """
    count = 0
    
    def __init__(self, bbox):
        """
        Initialize tracker with first detection.
        
        Parameters:
        bbox: [x1, y1, x2, y2] bounding box
        """
        # Initialize Kalman filter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                               [0, 1, 0, 0, 0, 1, 0],
                               [0, 0, 1, 0, 0, 0, 1],
                               [0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 0, 1]])
        
        # Measurement matrix
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0]])
        
        # Measurement noise
        self.kf.R[0:4, 0:4] *= 10
        
        # Process noise
        self.kf.Q[0:4, 0:4] *= 0.01
        self.kf.Q[4:7, 4:7] *= 0.01
        
        # Initial covariance
        self.kf.P[0:4, 0:4] *= 10
        self.kf.P[4:7, 4:7] *= 1000
        
        # Initialize state
        z = self.convert_bbox_to_z(bbox)
        self.kf.x = np.array([z[0], z[1], z[2], z[3], 0, 0, 0]).reshape((7, 1))
        
        # Tracker parameters
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        self.history = []
        self.age = 0
        self.time_since_update = 0
        self.hit_streak = 0
        
    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box.
        """
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        pred = self.convert_x_to_bbox(self.kf.x)
        self.history.append(pred)
        return pred
    
    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hit_streak += 1
        
        z = self.convert_bbox_to_z(bbox)
        self.kf.update(z)
    
    def get_state(self):
        """
        Return current bounding box estimate in [x1, y1, x2, y2] format.
        """
        return self.convert_x_to_bbox(self.kf.x)
    
    @staticmethod
    def convert_bbox_to_z(bbox):
        """
        Takes a bbox [x1, y1, x2, y2] and returns z = [cx, cy, w, h]
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        
        # Ensure dimensions are positive
        if w <= 0 or h <= 0:
            w = 1
            h = 1
            
        return np.array([cx, cy, w, h]).reshape((4, 1))
    
    @staticmethod
    def convert_x_to_bbox(x, score=None):
        """
        Takes state vector x and returns bbox in [x1, y1, x2, y2] format.
        """
        cx = x[0]
        cy = x[1]
        w = x[2]
        h = x[3]
        
        # Ensure width and height are positive
        if w < 1:
            w = 1
        if h < 1:
            h = 1
            
        x1 = int(cx - w / 2.0)
        y1 = int(cy - h / 2.0)
        x2 = int(cx + w / 2.0)
        y2 = int(cy + h / 2.0)
        
        return [x1, y1, x2, y2]


class Sort:
    """
    SORT tracker class for managing multiple objects.
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize SORT tracker.
        
        Parameters:
        max_age: Maximum number of frames to keep a track without detection
        min_hits: Minimum hits to return a track
        iou_threshold: IOU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        
    def update(self, dets):
        """
        Update tracker with new detections.
        
        Parameters:
        dets: List of detections [x1, y1, x2, y2, score, class]
        
        Returns:
        List of tracks [x1, y1, x2, y2, id]
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trk_pred = [pos[0], pos[1], pos[2], pos[3], trk.id]
            trks[t, :] = trk_pred
            
            # Remove trackers that are too old
            if trk.time_since_update > self.max_age:
                to_del.append(t)
        
        # Remove old trackers
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # Match detections to trackers
        if len(dets) > 0 and len(self.trackers) > 0:
            # Convert detections to format for IOU matching
            dets_boxes = np.array([[d[0], d[1], d[2], d[3]] for d in dets])
            
            # Calculate IOU matrix
            iou_matrix = np.zeros((len(dets), len(self.trackers)))
            for d, det in enumerate(dets_boxes):
                for t, trk in enumerate(trks):
                    iou_matrix[d, t] = iou(det, trk[:4])
            
            # Hungarian algorithm matching (simplified with greedy matching)
            matched_idx = []
            unmatched_dets = list(range(len(dets)))
            unmatched_trks = list(range(len(self.trackers)))
            
            # Greedy matching
            while len(matched_idx) < min(len(dets), len(self.trackers)):
                max_iou = np.max(iou_matrix)
                if max_iou < self.iou_threshold:
                    break
                d, t = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                matched_idx.append((d, t))
                iou_matrix[d, :] = -1
                iou_matrix[:, t] = -1
                if d in unmatched_dets:
                    unmatched_dets.remove(d)
                if t in unmatched_trks:
                    unmatched_trks.remove(t)
            
            # Update matched trackers
            for d, t in matched_idx:
                self.trackers[t].update(dets[d][:4])
                self.trackers[t].hit_streak += 1
            
            # Create new trackers for unmatched detections
            for d in unmatched_dets:
                trk = KalmanBoxTracker(dets[d][:4])
                self.trackers.append(trk)
        
        # Generate output
        for trk in self.trackers:
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append([d[0], d[1], d[2], d[3], trk.id])
        
        return np.array(ret) if len(ret) > 0 else np.empty((0, 5))
