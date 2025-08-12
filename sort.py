# sort.py
import numpy as np
from filterpy.kalman import KalmanFilter

def iou(bb_test, bb_gt):
    """
    Compute IoU between two bboxes in [x1,y1,x2,y2] format.
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union

class KalmanBoxTracker:
    """
    Represents the internal state of individual tracked objects observed as bounding boxes.
    State vector x: [cx, cy, s, r, vx, vy, vs]^T
      cx,cy = center
      s = scale (area)
      r = aspect ratio
      vx,vy,vs = velocities for cx,cy,s
    """
    count = 0

    def __init__(self, bbox):
        """
        bbox: [x1,y1,x2,y2]
        """
        # create kalman filter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # State transition
        self.kf.F = np.eye(7)
        self.kf.F[0,4] = 1.0  # cx += vx
        self.kf.F[1,5] = 1.0  # cy += vy
        self.kf.F[2,6] = 1.0  # s  += vs

        # Measurement function maps state to measurement space
        self.kf.H = np.zeros((4,7))
        self.kf.H[0,0] = 1.
        self.kf.H[1,1] = 1.
        self.kf.H[2,2] = 1.
        self.kf.H[3,3] = 1.

        # Covariances
        self.kf.R *= 1.0
        self.kf.P *= 10.0
        self.kf.Q *= 0.01

        # initialize state
        z = self.convert_bbox_to_z(bbox).reshape((4,))
        # x is (7,), set first 4 entries
        self.kf.x = np.zeros((7,1))
        self.kf.x[0,0] = z[0]
        self.kf.x[1,0] = z[1]
        self.kf.x[2,0] = z[2]
        self.kf.x[3,0] = z[3]

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Update the state vector with observed bbox.
        bbox: [x1,y1,x2,y2]
        """
        z = self.convert_bbox_to_z(bbox).reshape((4,))
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(z)

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

    def get_state(self):
        """
        Return current bounding box estimate in [x1,y1,x2,y2] format.
        """
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        """
        Takes a bbox [x1,y1,x2,y2] and returns z = [cx, cy, s, r]
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= 0 or h <= 0:
            # avoid invalid values
            return np.array([0., 0., 0., 0.]).reshape((4,1))
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h)
        return np.array([cx, cy, s, r]).reshape((4,1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        """
        Takes state variable x (shape (7,1) or (7,)) and returns bbox [x1,y1,x2,y2]
        """
        x = np.asarray(x).reshape((-1,))
        cx, cy, s, r = x[0], x[1], x[2], x[3]
        if s <= 0 or r <= 0:
            return np.array([0.,0.,0.,0.])
        w = np.sqrt(s * r)
        h = s / w
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return np.array([x1, y1, x2, y2])
