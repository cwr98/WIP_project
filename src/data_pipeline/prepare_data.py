"""Data preparation for DLR Aerial Crowd Dataset.

Steps:
1) Read raw imagery/labels from data/raw
2) Normalize/resize/patchify as needed
3) Export model-ready samples to data/processed
"""
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from src.utils.paths import RAW_DIR, PROCESSED_DIR, ensure_dirs

def find_raw_images():
    # Update the extensions to match your DLR files
    img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return [p for p in RAW_DIR.rglob("*") if p.suffix.lower() in img_exts]

def process_image(img_path: Path):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")
    # Example preprocessing: resize to 512x512 (adjust to your model)
    img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    return img_resized

def main():
    ensure_dirs()
    images = find_raw_images()
    out_dir = PROCESSED_DIR / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(images, desc="Preparing images"):
        try:
            arr = process_image(p)
            out_path = out_dir / (p.stem + "_512.png")
            cv2.imwrite(str(out_path), arr)
        except Exception as e:
            print(f"[WARN] {p}: {e}")

if __name__ == "__main__":
    main()
