from pathlib import Path

PROJECT_ROOT = Path("C:/Users/corba/WIP_project")
RAW_DIR = PROJECT_ROOT / "data/raw"

IMAGE_DIR = RAW_DIR / "DLR_AerialCrowdDataset" / "DLR_AerialCrowdDataset" / "Test" / "Images"
MASK_DIR  = RAW_DIR / "DLR_AerialCrowdDataset" / "DLR_AerialCrowdDataset" / "Test" / "Annotation"

def load_image_mask_pairs():
    images = {p.stem: p for p in IMAGE_DIR.glob("*.jpg")}
    masks  = {p.stem: p for p in MASK_DIR.glob("*.png")}

    pairs = {}
    for name, img_path in images.items():
        if name in masks:
            pairs[name] = {"image": str(img_path), "mask": str(masks[name])}
    return pairs

if __name__ == "__main__":
    pairs = load_image_mask_pairs()
    print(f"Found {len(pairs)} pairs")
    if pairs:
        for k, v in list(pairs.items())[:5]:
            print(k, "->", v)
            