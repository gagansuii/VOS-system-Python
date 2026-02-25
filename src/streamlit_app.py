from __future__ import annotations

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

from outfit_library import OutfitLibrary
from overlay import OverlayEngine
from pose import PoseEstimator
from config import DEFAULT_MODE, DEFAULT_UPPER_SCALE, DEFAULT_LOWER_SCALE


class VideoProcessor(VideoProcessorBase):
    def __init__(self, library: OutfitLibrary):
        self.library = library
        self.estimator = PoseEstimator()
        self.overlay = OverlayEngine()
        self.use_segmentation = True
        self.mode = DEFAULT_MODE
        self.upper_scale = DEFAULT_UPPER_SCALE
        self.lower_scale = DEFAULT_LOWER_SCALE
        self.debug = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        pose_res, seg_res = self.estimator.process(img)

        if pose_res.pose_landmarks:
            self.overlay.apply(
                img,
                pose_res.pose_landmarks,
                seg_res.segmentation_mask if seg_res else None,
                self.library.get_upper(),
                self.library.get_lower(),
                use_segmentation=self.use_segmentation,
                mode=self.mode,
                upper_scale=self.upper_scale,
                lower_scale=self.lower_scale,
                debug=self.debug,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.set_page_config(page_title="Virtual Outfit Generation", layout="wide")
st.title("Virtual Outfit Generation System")

with st.sidebar:
    outfit_root = st.text_input("Outfit folder", value="data/outfits")
    mode = st.selectbox("Mode", ["warp", "bbox"], index=0 if DEFAULT_MODE == "warp" else 1)
    use_segmentation = st.checkbox("Segmentation mask", value=True)
    upper_scale = st.slider("Upper scale", min_value=0.5, max_value=2.0, value=float(DEFAULT_UPPER_SCALE), step=0.05)
    lower_scale = st.slider("Lower scale", min_value=0.5, max_value=2.0, value=float(DEFAULT_LOWER_SCALE), step=0.05)
    debug = st.checkbox("Debug points", value=False)

    col1, col2 = st.columns(2)
    with col1:
        prev_upper = st.button("Prev upper")
        prev_lower = st.button("Prev lower")
    with col2:
        next_upper = st.button("Next upper")
        next_lower = st.button("Next lower")

if "outfit_root" not in st.session_state or st.session_state.outfit_root != outfit_root:
    st.session_state.outfit_root = outfit_root
    st.session_state.library = OutfitLibrary(outfit_root)

library: OutfitLibrary = st.session_state.library

if prev_upper:
    library.prev_upper()
if next_upper:
    library.next_upper()
if prev_lower:
    library.prev_lower()
if next_lower:
    library.next_lower()

st.caption(library.status())

ctx = webrtc_streamer(
    key="virtual-outfit",
    video_processor_factory=lambda: VideoProcessor(library),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.video_processor:
    ctx.video_processor.use_segmentation = use_segmentation
    ctx.video_processor.mode = mode
    ctx.video_processor.upper_scale = upper_scale
    ctx.video_processor.lower_scale = lower_scale
    ctx.video_processor.debug = debug
