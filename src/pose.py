from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import mediapipe as mp

from config import POSE_MIN_DET_CONF, POSE_MIN_TRACK_CONF, POSE_MODEL_COMPLEXITY

HAS_MP_SOLUTIONS = hasattr(mp, "solutions")


@dataclass
class SimpleLandmark:
    x: float
    y: float
    visibility: float = 1.0


@dataclass
class SimpleLandmarks:
    landmark: list


class PoseEstimator:
    def __init__(
        self,
        min_detection_conf=POSE_MIN_DET_CONF,
        min_tracking_conf=POSE_MIN_TRACK_CONF,
        model_complexity=POSE_MODEL_COMPLEXITY,
    ):
        self.use_mp = HAS_MP_SOLUTIONS
        if self.use_mp:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_conf,
                min_tracking_confidence=min_tracking_conf,
            )
            self.segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        else:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def _hog_landmarks(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = 1.0
        if w > 720:
            scale = 720.0 / w
            frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
        rects, _ = self.hog.detectMultiScale(
            frame_bgr,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        if len(rects) == 0:
            return None

        x, y, rw, rh = max(rects, key=lambda r: r[2] * r[3])
        if scale != 1.0:
            x = int(x / scale)
            y = int(y / scale)
            rw = int(rw / scale)
            rh = int(rh / scale)

        def norm(px, py):
            return SimpleLandmark(px / w, py / h, 1.0)

        landmarks = [SimpleLandmark(0.0, 0.0, 0.0) for _ in range(33)]

        # Heuristic body points within the detected person box
        ls = (x + 0.30 * rw, y + 0.25 * rh)
        rs = (x + 0.70 * rw, y + 0.25 * rh)
        lh = (x + 0.40 * rw, y + 0.55 * rh)
        rhp = (x + 0.60 * rw, y + 0.55 * rh)
        lk = (x + 0.45 * rw, y + 0.75 * rh)
        rk = (x + 0.55 * rw, y + 0.75 * rh)
        la = (x + 0.45 * rw, y + 0.95 * rh)
        ra = (x + 0.55 * rw, y + 0.95 * rh)

        landmarks[11] = norm(*ls)
        landmarks[12] = norm(*rs)
        landmarks[23] = norm(*lh)
        landmarks[24] = norm(*rhp)
        landmarks[25] = norm(*lk)
        landmarks[26] = norm(*rk)
        landmarks[27] = norm(*la)
        landmarks[28] = norm(*ra)

        return SimpleLandmarks(landmark=landmarks)

    def process(self, frame_bgr):
        if self.use_mp:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pose_result = self.pose.process(frame_rgb)
            seg_result = self.segmenter.process(frame_rgb)
            return pose_result, seg_result

        landmarks = self._hog_landmarks(frame_bgr)
        pose_result = SimpleNamespace(pose_landmarks=landmarks)
        seg_result = SimpleNamespace(segmentation_mask=None)
        return pose_result, seg_result
