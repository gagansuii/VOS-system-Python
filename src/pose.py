from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import time

import cv2
import numpy as np
import mediapipe as mp

from config import (
    MODELS_DIR,
    POSE_MIN_DET_CONF,
    POSE_MIN_TRACK_CONF,
    POSE_MODEL_COMPLEXITY,
    POSE_MODEL_NAMES,
    POSE_MODEL_PREFERRED,
    SEGMENTER_MODEL_NAME,
)

HAS_MP_SOLUTIONS = hasattr(mp, "solutions")
try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    HAS_MP_TASKS = True
except Exception:
    mp_python = None
    vision = None
    HAS_MP_TASKS = False


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
        self.backend = "hog"
        self.segmentation_backend = "none"
        self.warnings = []
        self._clock = time.monotonic()

        self.pose = None
        self.segmenter = None

        # Prefer MediaPipe Tasks when available
        if HAS_MP_TASKS:
            pose_model_path = self._select_pose_model()
            seg_model_path = MODELS_DIR / SEGMENTER_MODEL_NAME

            if pose_model_path:
                pose_options = vision.PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(pose_model_path)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=min_detection_conf,
                    min_pose_presence_confidence=min_detection_conf,
                    min_tracking_confidence=min_tracking_conf,
                )
                self.pose = vision.PoseLandmarker.create_from_options(pose_options)
                self.backend = "tasks"
            else:
                self.warnings.append("Pose model not found. Run scripts/download_models.py")

            if seg_model_path.exists():
                seg_options = vision.ImageSegmenterOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(seg_model_path)),
                    running_mode=vision.RunningMode.VIDEO,
                    output_category_mask=True,
                    output_confidence_masks=False,
                )
                self.segmenter = vision.ImageSegmenter.create_from_options(seg_options)
                self.segmentation_backend = "tasks"
            else:
                self.warnings.append("Segmentation model not found. Run scripts/download_models.py")

        if self.backend == "tasks":
            return

        if HAS_MP_SOLUTIONS:
            self.backend = "solutions"
            self.segmentation_backend = "solutions"
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_conf,
                min_tracking_confidence=min_tracking_conf,
            )
            self.segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
            return

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def _select_pose_model(self):
        preferred = POSE_MODEL_PREFERRED
        variants = [preferred, "full"] if preferred != "full" else ["full"]
        for variant in variants:
            name = POSE_MODEL_NAMES.get(variant)
            if not name:
                continue
            path = MODELS_DIR / name
            if path.exists():
                return path
        return None

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
        if self.backend == "tasks":
            ts_ms = int((time.monotonic() - self._clock) * 1000)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            pose_res = self.pose.detect_for_video(mp_image, ts_ms) if self.pose else None
            seg_res = self.segmenter.segment_for_video(mp_image, ts_ms) if self.segmenter else None

            pose_landmarks = None
            if pose_res and pose_res.pose_landmarks:
                lm_list = pose_res.pose_landmarks[0]
                landmarks = [
                    SimpleLandmark(lm.x, lm.y, getattr(lm, "visibility", 1.0)) for lm in lm_list
                ]
                pose_landmarks = SimpleLandmarks(landmark=landmarks)

            seg_mask = None
            if seg_res and getattr(seg_res, "category_mask", None) is not None:
                cat_mask = seg_res.category_mask
                if hasattr(cat_mask, "numpy_view"):
                    mask_np = cat_mask.numpy_view()
                else:
                    mask_np = np.array(cat_mask)
                if mask_np.ndim == 3:
                    mask_np = mask_np[:, :, 0]
                h, w = frame_bgr.shape[:2]
                if mask_np.shape[0] != h or mask_np.shape[1] != w:
                    mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                seg_mask = (mask_np > 0).astype(np.float32)

            pose_result = SimpleNamespace(pose_landmarks=pose_landmarks)
            seg_result = SimpleNamespace(segmentation_mask=seg_mask)
            return pose_result, seg_result

        if self.backend == "solutions":
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pose_result = self.pose.process(frame_rgb)
            seg_result = self.segmenter.process(frame_rgb)
            return pose_result, seg_result

        landmarks = self._hog_landmarks(frame_bgr)
        pose_result = SimpleNamespace(pose_landmarks=landmarks)
        seg_result = SimpleNamespace(segmentation_mask=None)
        return pose_result, seg_result

    def status(self) -> str:
        msg = f"Pose backend: {self.backend} | Segmentation backend: {self.segmentation_backend}"
        if self.warnings:
            msg += " | " + " | ".join(self.warnings)
        return msg
