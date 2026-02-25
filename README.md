# Virtual Outfit Generation System

This is an initial scaffold for a real-time virtual outfit overlay system using OpenCV and MediaPipe.

## Quick start

1. Create a virtual environment (optional):
   - `python -m venv .venv`
   - `.venv\Scripts\activate`
2. Install dependencies:
   - `python -m pip install -r requirements.txt`
3. Add outfit images:
   - Upper garments in `data/outfits/upper/`
   - Lower garments in `data/outfits/lower/`
   - Use PNGs with transparent backgrounds for best results.
4. Run:
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

- This uses MediaPipe Pose + Selfie Segmentation for a lightweight real-time pipeline.
- Alignment is heuristic and will need tuning for different body types and garments.
- For higher fidelity, replace the overlay with a pose-guided warp model (future work).
