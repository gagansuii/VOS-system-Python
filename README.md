# Virtual Outfit Generation System

This is an initial scaffold for a real-time virtual outfit overlay system using OpenCV and MediaPipe.

## Quick start

1. Create a virtual environment (optional):
   - `python -m venv .venv`
   - `.venv\Scripts\activate`
2. Install dependencies:
   - `python -m pip install -r requirements.txt`
3. Download MediaPipe Tasks models (recommended for accuracy):
   - `python scripts/download_models.py`
   - This prefers the heavy pose model and falls back to full if needed.
4. Add outfit images:
   - Upper garments in `data/outfits/upper/`
   - Lower garments in `data/outfits/lower/`
   - Use PNGs with transparent backgrounds for best results.
5. Run:
   - `python src/main.py --camera 0`

If the camera window is black, try a different backend:
- `python src/main.py --camera 0 --backend dshow`
- `python src/main.py --camera 0 --backend msmf`

## Streamlit app

Run a web UI with camera access:

- `streamlit run src/streamlit_app.py`

## Controls

- `u` / `j`: next / previous upper garment
- `l` / `k`: next / previous lower garment
- `s`: toggle segmentation mask blending
- `m`: toggle mode (bbox/warp)
- `+` / `-`: increase / decrease upper scale
- `]` / `[` : increase / decrease lower scale
- `r`: reset scales
- `d`: toggle debug points
- `h`: show help in console
- `q`: quit

## Notes

- With models downloaded, this uses MediaPipe Tasks Pose Landmarker + Image Segmenter.
- If models are missing, it falls back to a lightweight HOG-based detector (no true segmentation).
- Alignment is heuristic and will need tuning for different body types and garments.
- For higher fidelity, replace the overlay with a pose-guided warp model (future work).
