from __future__ import annotations

import argparse
import sys
import time

import cv2

from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    DEFAULT_MODE,
    DEFAULT_UPPER_SCALE,
    DEFAULT_LOWER_SCALE,
)
from outfit_library import OutfitLibrary
from overlay import OverlayEngine
from pose import PoseEstimator


def parse_args():
    parser = argparse.ArgumentParser(description="Virtual Outfit Generation System")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index")
    parser.add_argument("--outfits", type=str, default="data/outfits", help="Outfit folder root")
    parser.add_argument("--no-segmentation", action="store_true", help="Disable segmentation mask blending")
    parser.add_argument("--backend", choices=["auto", "dshow", "msmf", "any"], default="auto", help="Camera backend")
    parser.add_argument("--mode", choices=["bbox", "warp"], default=DEFAULT_MODE, help="Overlay mode")
    parser.add_argument("--upper-scale", type=float, default=DEFAULT_UPPER_SCALE, help="Upper garment scale")
    parser.add_argument("--lower-scale", type=float, default=DEFAULT_LOWER_SCALE, help="Lower garment scale")
    parser.add_argument("--debug", action="store_true", help="Draw debug points")
    return parser.parse_args()


def print_help():
    print("\nControls:")
    print("  u / j : next / previous upper garment")
    print("  l / k : next / previous lower garment")
    print("  s     : toggle segmentation blending")
    print("  m     : toggle mode (bbox/warp)")
    print("  + / - : increase / decrease upper scale")
    print("  ] / [ : increase / decrease lower scale")
    print("  r     : reset scales to defaults")
    print("  d     : toggle debug points")
    print("  h     : show this help")
    print("  q     : quit\n")


def open_camera(index: int, backend: str):
    backend_map = {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "any": cv2.CAP_ANY,
    }
    if backend != "auto":
        cap = cv2.VideoCapture(index, backend_map[backend])
        return cap, backend

    candidates = []
    if sys.platform.startswith("win"):
        candidates = ["dshow", "msmf", "any"]
    else:
        candidates = ["any"]

    for name in candidates:
        cap = cv2.VideoCapture(index, backend_map[name])
        if cap.isOpened():
            # Warm up
            for _ in range(5):
                cap.read()
                time.sleep(0.01)
            return cap, name
        cap.release()

    cap = cv2.VideoCapture(index)
    return cap, "auto"


def main():
    args = parse_args()
    outfits = OutfitLibrary(args.outfits)
    print(outfits.status())

    cap, backend_used = open_camera(args.camera, args.backend)
    if not cap.isOpened():
        print("Failed to open camera", file=sys.stderr)
        return 1
    print(f"Camera backend: {backend_used}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    estimator = PoseEstimator()
    print(estimator.status())
    overlay = OverlayEngine()
    use_segmentation = not args.no_segmentation
    mode = args.mode
    upper_scale = args.upper_scale
    lower_scale = args.lower_scale
    debug = args.debug

    print_help()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        pose_res, seg_res = estimator.process(frame)

        if pose_res.pose_landmarks:
            overlay.apply(
                frame,
                pose_res.pose_landmarks,
                seg_res.segmentation_mask if seg_res else None,
                outfits.get_upper(),
                outfits.get_lower(),
                use_segmentation=use_segmentation,
                mode=mode,
                upper_scale=upper_scale,
                lower_scale=lower_scale,
                debug=debug,
            )

        cv2.imshow("Virtual Outfit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("u"):
            outfits.next_upper()
        elif key == ord("j"):
            outfits.prev_upper()
        elif key == ord("l"):
            outfits.next_lower()
        elif key == ord("k"):
            outfits.prev_lower()
        elif key == ord("s"):
            use_segmentation = not use_segmentation
            print(f"Segmentation: {'ON' if use_segmentation else 'OFF'}")
        elif key == ord("m"):
            mode = "warp" if mode == "bbox" else "bbox"
            print(f"Mode: {mode}")
        elif key in (ord("+"), ord("=")):
            upper_scale = min(2.0, upper_scale + 0.05)
            print(f"Upper scale: {upper_scale:.2f}")
        elif key == ord("-"):
            upper_scale = max(0.5, upper_scale - 0.05)
            print(f"Upper scale: {upper_scale:.2f}")
        elif key == ord("]"):
            lower_scale = min(2.0, lower_scale + 0.05)
            print(f"Lower scale: {lower_scale:.2f}")
        elif key == ord("["):
            lower_scale = max(0.5, lower_scale - 0.05)
            print(f"Lower scale: {lower_scale:.2f}")
        elif key == ord("r"):
            upper_scale = DEFAULT_UPPER_SCALE
            lower_scale = DEFAULT_LOWER_SCALE
            print("Scales reset")
        elif key == ord("d"):
            debug = not debug
            print(f"Debug: {'ON' if debug else 'OFF'}")
        elif key == ord("h"):
            print_help()

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
