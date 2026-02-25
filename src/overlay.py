from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    _MP_POSE = mp.solutions.pose.PoseLandmark if hasattr(mp, "solutions") else None
except Exception:
    _MP_POSE = None


class _LmEnum:
    def __init__(self, value: int):
        self.value = value


class _FallbackPoseLandmark:
    LEFT_SHOULDER = _LmEnum(11)
    RIGHT_SHOULDER = _LmEnum(12)
    LEFT_HIP = _LmEnum(23)
    RIGHT_HIP = _LmEnum(24)
    LEFT_KNEE = _LmEnum(25)
    RIGHT_KNEE = _LmEnum(26)
    LEFT_ANKLE = _LmEnum(27)
    RIGHT_ANKLE = _LmEnum(28)

from config import (
    VISIBILITY_THRESHOLD,
    SEGMENTATION_THRESHOLD,
    UPPER_WIDTH_PAD,
    UPPER_HEIGHT_PAD_TOP,
    UPPER_HEIGHT_PAD_BOTTOM,
    LOWER_WIDTH_PAD,
    LOWER_HEIGHT_PAD_TOP,
    LOWER_HEIGHT_PAD_BOTTOM,
    UPPER_SRC_LEFT_SHOULDER,
    UPPER_SRC_RIGHT_SHOULDER,
    UPPER_SRC_LEFT_HIP,
    LOWER_SRC_LEFT_HIP,
    LOWER_SRC_RIGHT_HIP,
    LOWER_SRC_LEFT_ANKLE,
    UPPER_SHOULDER_OFFSET,
    UPPER_HIP_OFFSET,
    LOWER_HIP_OFFSET,
    LOWER_ANKLE_OFFSET,
)


@dataclass
class OverlayEngine:
    visibility_threshold: float = VISIBILITY_THRESHOLD
    segmentation_threshold: float = SEGMENTATION_THRESHOLD

    def _lm(self, landmarks, lm_enum):
        try:
            lm = landmarks.landmark[lm_enum.value]
        except Exception:
            return None
        if lm.visibility < self.visibility_threshold:
            return None
        return lm

    def _to_pixel(self, lm, w, h) -> np.ndarray:
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)

    def _normalize(self, v: np.ndarray) -> Optional[np.ndarray]:
        n = np.linalg.norm(v)
        if n < 1e-6:
            return None
        return v / n

    def _orthonormal_y(self, x_axis: np.ndarray, y_axis: Optional[np.ndarray]) -> np.ndarray:
        if y_axis is None:
            return np.array([-x_axis[1], x_axis[0]], dtype=np.float32)
        y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
        n = np.linalg.norm(y_axis)
        if n < 1e-6:
            return np.array([-x_axis[1], x_axis[0]], dtype=np.float32)
        return y_axis / n

    def _bbox_from_points(
        self,
        xs,
        ys,
        pad_x,
        pad_y_top,
        pad_y_bottom,
    ) -> Optional[Tuple[float, float, float, float]]:
        if not xs or not ys:
            return None
        x1 = min(xs)
        x2 = max(xs)
        y1 = min(ys)
        y2 = max(ys)
        w = max(1e-6, x2 - x1)
        h = max(1e-6, y2 - y1)
        x1 -= w * pad_x
        x2 += w * pad_x
        y1 -= h * pad_y_top
        y2 += h * pad_y_bottom
        return x1, y1, x2, y2

    def upper_bbox(self, landmarks) -> Optional[Tuple[float, float, float, float]]:
        mp_pose = _MP_POSE or _FallbackPoseLandmark
        ls = self._lm(landmarks, mp_pose.LEFT_SHOULDER)
        rs = self._lm(landmarks, mp_pose.RIGHT_SHOULDER)
        lh = self._lm(landmarks, mp_pose.LEFT_HIP)
        rh = self._lm(landmarks, mp_pose.RIGHT_HIP)
        if not all([ls, rs, lh, rh]):
            return None
        xs = [ls.x, rs.x]
        ys = [ls.y, rs.y, lh.y, rh.y]
        return self._bbox_from_points(xs, ys, UPPER_WIDTH_PAD, UPPER_HEIGHT_PAD_TOP, UPPER_HEIGHT_PAD_BOTTOM)

    def lower_bbox(self, landmarks) -> Optional[Tuple[float, float, float, float]]:
        mp_pose = _MP_POSE or _FallbackPoseLandmark
        lh = self._lm(landmarks, mp_pose.LEFT_HIP)
        rh = self._lm(landmarks, mp_pose.RIGHT_HIP)
        lk = self._lm(landmarks, mp_pose.LEFT_KNEE)
        rk = self._lm(landmarks, mp_pose.RIGHT_KNEE)
        la = self._lm(landmarks, mp_pose.LEFT_ANKLE)
        ra = self._lm(landmarks, mp_pose.RIGHT_ANKLE)
        if not all([lh, rh]):
            return None
        leg_pts = [p for p in [lk, rk, la, ra] if p is not None]
        if not leg_pts:
            return None
        xs = [lh.x, rh.x]
        ys = [lh.y, rh.y] + [p.y for p in leg_pts]
        return self._bbox_from_points(xs, ys, LOWER_WIDTH_PAD, LOWER_HEIGHT_PAD_TOP, LOWER_HEIGHT_PAD_BOTTOM)

    def _src_points_upper(self, outfit) -> np.ndarray:
        h, w = outfit.shape[:2]
        return np.array(
            [
                [w * UPPER_SRC_LEFT_SHOULDER[0], h * UPPER_SRC_LEFT_SHOULDER[1]],
                [w * UPPER_SRC_RIGHT_SHOULDER[0], h * UPPER_SRC_RIGHT_SHOULDER[1]],
                [w * UPPER_SRC_LEFT_HIP[0], h * UPPER_SRC_LEFT_HIP[1]],
            ],
            dtype=np.float32,
        )

    def _src_points_lower(self, outfit) -> np.ndarray:
        h, w = outfit.shape[:2]
        return np.array(
            [
                [w * LOWER_SRC_LEFT_HIP[0], h * LOWER_SRC_LEFT_HIP[1]],
                [w * LOWER_SRC_RIGHT_HIP[0], h * LOWER_SRC_RIGHT_HIP[1]],
                [w * LOWER_SRC_LEFT_ANKLE[0], h * LOWER_SRC_LEFT_ANKLE[1]],
            ],
            dtype=np.float32,
        )

    def _upper_target_points(self, landmarks, w, h, scale: float) -> Optional[np.ndarray]:
        mp_pose = _MP_POSE or _FallbackPoseLandmark
        ls = self._lm(landmarks, mp_pose.LEFT_SHOULDER)
        rs = self._lm(landmarks, mp_pose.RIGHT_SHOULDER)
        lh = self._lm(landmarks, mp_pose.LEFT_HIP)
        rh = self._lm(landmarks, mp_pose.RIGHT_HIP)
        if not all([ls, rs, lh, rh]):
            return None

        ls_p = self._to_pixel(ls, w, h)
        rs_p = self._to_pixel(rs, w, h)
        lh_p = self._to_pixel(lh, w, h)
        rh_p = self._to_pixel(rh, w, h)

        mid_sh = (ls_p + rs_p) * 0.5
        mid_hip = (lh_p + rh_p) * 0.5

        x_axis = self._normalize(rs_p - ls_p)
        if x_axis is None:
            return None
        y_axis = self._normalize(mid_hip - mid_sh)
        y_axis = self._orthonormal_y(x_axis, y_axis)

        width = np.linalg.norm(rs_p - ls_p)
        height = np.linalg.norm(mid_hip - mid_sh)
        if height < 1e-6:
            return None

        half_w = 0.5 * width * scale
        height_scaled = height * scale

        left_sh = mid_sh - x_axis * half_w + y_axis * (height_scaled * UPPER_SHOULDER_OFFSET)
        right_sh = mid_sh + x_axis * half_w + y_axis * (height_scaled * UPPER_SHOULDER_OFFSET)
        left_hip = mid_sh - x_axis * half_w + y_axis * (height_scaled * UPPER_HIP_OFFSET)

        return np.vstack([left_sh, right_sh, left_hip]).astype(np.float32)

    def _lower_target_points(self, landmarks, w, h, scale: float) -> Optional[np.ndarray]:
        mp_pose = _MP_POSE or _FallbackPoseLandmark
        lh = self._lm(landmarks, mp_pose.LEFT_HIP)
        rh = self._lm(landmarks, mp_pose.RIGHT_HIP)
        lk = self._lm(landmarks, mp_pose.LEFT_KNEE)
        rk = self._lm(landmarks, mp_pose.RIGHT_KNEE)
        la = self._lm(landmarks, mp_pose.LEFT_ANKLE)
        ra = self._lm(landmarks, mp_pose.RIGHT_ANKLE)
        if not all([lh, rh]):
            return None

        lh_p = self._to_pixel(lh, w, h)
        rh_p = self._to_pixel(rh, w, h)

        ankle_pts = [p for p in [la, ra] if p is not None]
        knee_pts = [p for p in [lk, rk] if p is not None]

        if ankle_pts:
            mid_ankle = sum(self._to_pixel(p, w, h) for p in ankle_pts) / len(ankle_pts)
        elif knee_pts:
            mid_ankle = sum(self._to_pixel(p, w, h) for p in knee_pts) / len(knee_pts)
        else:
            return None

        mid_hip = (lh_p + rh_p) * 0.5
        x_axis = self._normalize(rh_p - lh_p)
        if x_axis is None:
            return None
        y_axis = self._normalize(mid_ankle - mid_hip)
        y_axis = self._orthonormal_y(x_axis, y_axis)

        width = np.linalg.norm(rh_p - lh_p)
        height = np.linalg.norm(mid_ankle - mid_hip)
        if height < 1e-6:
            return None

        half_w = 0.5 * width * scale
        height_scaled = height * scale

        left_hip = mid_hip - x_axis * half_w + y_axis * (height_scaled * LOWER_HIP_OFFSET)
        right_hip = mid_hip + x_axis * half_w + y_axis * (height_scaled * LOWER_HIP_OFFSET)
        left_ankle = mid_hip - x_axis * half_w + y_axis * (height_scaled * LOWER_ANKLE_OFFSET)

        return np.vstack([left_hip, right_hip, left_ankle]).astype(np.float32)

    def _overlay(self, frame, outfit, bbox, segmentation_mask=None):
        if outfit is None or bbox is None:
            return
        if outfit.ndim == 2:
            outfit = cv2.cvtColor(outfit, cv2.COLOR_GRAY2BGR)
        if outfit.ndim != 3 or outfit.shape[2] not in (3, 4):
            return

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = int(x1 * w)
        x2 = int(x2 * w)
        y1 = int(y1 * h)
        y2 = int(y2 * h)

        if x2 <= x1 or y2 <= y1:
            return

        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(w, x2)
        y2c = min(h, y2)
        if x2c <= x1c or y2c <= y1c:
            return

        out_w = x2 - x1
        out_h = y2 - y1
        if out_w <= 0 or out_h <= 0:
            return

        outfit_resized = cv2.resize(outfit, (out_w, out_h), interpolation=cv2.INTER_AREA)
        ox1 = x1c - x1
        oy1 = y1c - y1
        ox2 = ox1 + (x2c - x1c)
        oy2 = oy1 + (y2c - y1c)
        outfit_crop = outfit_resized[oy1:oy2, ox1:ox2]

        if outfit_crop.shape[2] == 4:
            alpha = outfit_crop[:, :, 3] / 255.0
            outfit_rgb = outfit_crop[:, :, :3]
        else:
            outfit_rgb = outfit_crop
            gray = cv2.cvtColor(outfit_rgb, cv2.COLOR_BGR2GRAY)
            alpha = (gray > 5).astype(np.float32)

        if segmentation_mask is not None:
            seg_roi = segmentation_mask[y1c:y2c, x1c:x2c]
            alpha = alpha * seg_roi

        alpha = np.clip(alpha, 0.0, 1.0)
        alpha_3 = np.dstack([alpha] * 3)

        roi = frame[y1c:y2c, x1c:x2c]
        blended = roi * (1 - alpha_3) + outfit_rgb * alpha_3
        frame[y1c:y2c, x1c:x2c] = blended.astype(np.uint8)

    def _warp_overlay(self, frame, outfit, src_pts, dst_pts, segmentation_mask=None):
        if outfit is None or src_pts is None or dst_pts is None:
            return
        if outfit.ndim == 2:
            outfit = cv2.cvtColor(outfit, cv2.COLOR_GRAY2BGR)
        if outfit.ndim != 3 or outfit.shape[2] not in (3, 4):
            return

        h, w = frame.shape[:2]
        M = cv2.getAffineTransform(src_pts, dst_pts)
        border_value = (0, 0, 0, 0) if outfit.shape[2] == 4 else (0, 0, 0)
        warped = cv2.warpAffine(
            outfit,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )

        if outfit.shape[2] == 4:
            alpha = warped[:, :, 3] / 255.0
            outfit_rgb = warped[:, :, :3]
        else:
            outfit_rgb = warped
            gray = cv2.cvtColor(outfit_rgb, cv2.COLOR_BGR2GRAY)
            alpha = (gray > 5).astype(np.float32)

        if segmentation_mask is not None:
            alpha = alpha * segmentation_mask

        alpha = np.clip(alpha, 0.0, 1.0)
        alpha_3 = np.dstack([alpha] * 3)

        blended = frame * (1 - alpha_3) + outfit_rgb * alpha_3
        frame[:, :, :] = blended.astype(np.uint8)

    def _draw_debug(self, frame, pts, color):
        for p in pts:
            cv2.circle(frame, (int(p[0]), int(p[1])), 4, color, -1)

    def apply(
        self,
        frame,
        landmarks,
        segmentation_mask,
        upper_img=None,
        lower_img=None,
        use_segmentation=True,
        mode="bbox",
        upper_scale=1.0,
        lower_scale=1.0,
        debug=False,
    ):
        seg = None
        if use_segmentation and segmentation_mask is not None:
            seg = (segmentation_mask > self.segmentation_threshold).astype(np.float32)

        h, w = frame.shape[:2]

        if mode == "warp":
            upper_src = self._src_points_upper(upper_img) if upper_img is not None else None
            upper_dst = self._upper_target_points(landmarks, w, h, upper_scale)
            if upper_src is not None and upper_dst is not None:
                self._warp_overlay(frame, upper_img, upper_src, upper_dst, seg)
                if debug:
                    self._draw_debug(frame, upper_dst, (0, 255, 0))
            else:
                self._overlay(frame, upper_img, self.upper_bbox(landmarks), seg)

            lower_src = self._src_points_lower(lower_img) if lower_img is not None else None
            lower_dst = self._lower_target_points(landmarks, w, h, lower_scale)
            if lower_src is not None and lower_dst is not None:
                self._warp_overlay(frame, lower_img, lower_src, lower_dst, seg)
                if debug:
                    self._draw_debug(frame, lower_dst, (255, 0, 0))
            else:
                self._overlay(frame, lower_img, self.lower_bbox(landmarks), seg)
        else:
            self._overlay(frame, upper_img, self.upper_bbox(landmarks), seg)
            self._overlay(frame, lower_img, self.lower_bbox(landmarks), seg)
