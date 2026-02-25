from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".avif"}
UPPER_KEYWORDS = ["shirt", "tshirt", "tee", "hoodie", "vest", "cap", "sweatshirt", "jacket", "coat"]
LOWER_KEYWORDS = ["pant", "pants", "short", "shorts", "jean", "skirt"]


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def _categorize(name: str) -> Optional[str]:
    low = name.lower()
    if any(k in low for k in UPPER_KEYWORDS):
        return "upper"
    if any(k in low for k in LOWER_KEYWORDS):
        return "lower"
    return None


@dataclass
class OutfitLibrary:
    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.upper_paths: List[Path] = []
        self.lower_paths: List[Path] = []
        self._cache = {}
        self._load()
        self.upper_idx = 0
        self.lower_idx = 0

    def _collect_from_dir(self, folder: Path) -> List[Path]:
        if not folder.exists():
            return []
        return sorted([p for p in folder.iterdir() if p.is_file() and _is_image(p)])

    def _load(self) -> None:
        upper_dir = self.root_dir / "upper"
        lower_dir = self.root_dir / "lower"

        self.upper_paths.extend(self._collect_from_dir(upper_dir))
        self.lower_paths.extend(self._collect_from_dir(lower_dir))

        if self.upper_paths or self.lower_paths:
            return

        # Fallback: auto-categorize from root
        if not self.root_dir.exists():
            return
        for p in self.root_dir.iterdir():
            if not p.is_file() or not _is_image(p):
                continue
            category = _categorize(p.name)
            if category == "upper":
                self.upper_paths.append(p)
            elif category == "lower":
                self.lower_paths.append(p)

    def _load_image(self, path: Path):
        if path in self._cache:
            return self._cache[path]
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            img = self._load_with_pil(path)
        self._cache[path] = img
        return img

    def _load_with_pil(self, path: Path):
        try:
            im = Image.open(path)
        except Exception:
            return None

        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            arr = np.array(im)
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)

        im = im.convert("RGB")
        arr = np.array(im)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def get_upper(self):
        if not self.upper_paths:
            return None
        return self._load_image(self.upper_paths[self.upper_idx])

    def get_lower(self):
        if not self.lower_paths:
            return None
        return self._load_image(self.lower_paths[self.lower_idx])

    def next_upper(self):
        if self.upper_paths:
            self.upper_idx = (self.upper_idx + 1) % len(self.upper_paths)

    def prev_upper(self):
        if self.upper_paths:
            self.upper_idx = (self.upper_idx - 1) % len(self.upper_paths)

    def next_lower(self):
        if self.lower_paths:
            self.lower_idx = (self.lower_idx + 1) % len(self.lower_paths)

    def prev_lower(self):
        if self.lower_paths:
            self.lower_idx = (self.lower_idx - 1) % len(self.lower_paths)

    def status(self) -> str:
        return (
            f"Upper: {len(self.upper_paths)} item(s), index {self.upper_idx} | "
            f"Lower: {len(self.lower_paths)} item(s), index {self.lower_idx}"
        )
