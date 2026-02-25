from __future__ import annotations

import argparse
from pathlib import Path
import sys
from urllib.error import URLError
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import (
    MODELS_DIR,
    POSE_MODEL_NAMES,
    POSE_MODEL_PREFERRED,
    POSE_MODEL_URLS,
    SEGMENTER_MODEL_NAME,
    SEGMENTER_MODEL_URL,
)


def _download(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"Exists: {dest.name}")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name}...")
    try:
        urlretrieve(url, dest)
        print(f"Saved: {dest}")
        return True
    except URLError as exc:
        print(f"Failed: {dest.name} ({exc})")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download MediaPipe Tasks models")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all pose variants (full + heavy)",
    )
    args = parser.parse_args()

    if args.all:
        pose_variants = list(POSE_MODEL_NAMES.keys())
        for variant in pose_variants:
            name = POSE_MODEL_NAMES[variant]
            url = POSE_MODEL_URLS[variant]
            _download(url, MODELS_DIR / name)
    else:
        preferred = POSE_MODEL_PREFERRED
        name = POSE_MODEL_NAMES[preferred]
        url = POSE_MODEL_URLS[preferred]
        ok = _download(url, MODELS_DIR / name)
        if not ok and preferred != "full":
            name = POSE_MODEL_NAMES["full"]
            url = POSE_MODEL_URLS["full"]
            _download(url, MODELS_DIR / name)

    _download(SEGMENTER_MODEL_URL, MODELS_DIR / SEGMENTER_MODEL_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
