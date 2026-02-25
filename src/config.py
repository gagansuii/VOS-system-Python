# Configuration defaults

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
POSE_MIN_DET_CONF = 0.5
POSE_MIN_TRACK_CONF = 0.5
POSE_MODEL_COMPLEXITY = 1
POSE_MODEL_PREFERRED = "heavy"  # "full" or "heavy" if available
VISIBILITY_THRESHOLD = 0.5
SEGMENTATION_THRESHOLD = 0.2
DEFAULT_MODE = "warp"  # "warp" or "bbox"
DEFAULT_UPPER_SCALE = 1.0
DEFAULT_LOWER_SCALE = 1.0
UPPER_WIDTH_PAD = 0.25
UPPER_HEIGHT_PAD_TOP = 0.25
UPPER_HEIGHT_PAD_BOTTOM = 0.10
LOWER_WIDTH_PAD = 0.25
LOWER_HEIGHT_PAD_TOP = 0.05
LOWER_HEIGHT_PAD_BOTTOM = 0.10

# Outfit anchor points in source images (fractions of width/height)
UPPER_SRC_LEFT_SHOULDER = (0.20, 0.18)
UPPER_SRC_RIGHT_SHOULDER = (0.80, 0.18)
UPPER_SRC_LEFT_HIP = (0.20, 0.96)

LOWER_SRC_LEFT_HIP = (0.20, 0.05)
LOWER_SRC_RIGHT_HIP = (0.80, 0.05)
LOWER_SRC_LEFT_ANKLE = (0.20, 0.96)

# Body-relative offsets for warp targets
UPPER_SHOULDER_OFFSET = -0.05
UPPER_HIP_OFFSET = 1.05
LOWER_HIP_OFFSET = 0.02
LOWER_ANKLE_OFFSET = 1.05

# MediaPipe Tasks model assets
POSE_MODEL_NAMES = {
    "full": "pose_landmarker_full.task",
    "heavy": "pose_landmarker_heavy.task",
}
POSE_MODEL_URLS = {
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
SEGMENTER_MODEL_NAME = "selfie_multiclass_256x256.tflite"
SEGMENTER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
