# setup_project.ps1
# Creates a clean ML project structure for the DLR Aerial Crowd project.

$ROOT = "C:\Users\corba\WIP_project"

$dirs = @(
  "$ROOT\data",
  "$ROOT\data\raw",
  "$ROOT\data\interim",
  "$ROOT\data\processed",
  "$ROOT\data\external",
  "$ROOT\notebooks",
  "$ROOT\src",
  "$ROOT\src\data_pipeline",
  "$ROOT\src\models",
  "$ROOT\src\utils",
  "$ROOT\experiments",
  "$ROOT\models",
  "$ROOT\reports\figures",
  "$ROOT\webapp"
)

# Create directories (idempotent)
foreach ($d in $dirs) { New-Item -ItemType Directory -Path $d -Force | Out-Null }

# --- Root files ---
$readme = @"
# DLR Aerial Crowd – Event Detection

This repo contains a geospatial ML pipeline using the **DLR Aerial Crowd Dataset**.

## Quick start
1. Put the original dataset archive(s) in `data/raw/` and extract them **there**.
2. Create a virtual environment:
   \`python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt\`
3. Run the data prep:
   \`python -m src.data_pipeline.prepare_data\`
4. Train:
   \`python -m src.models.train_model\`
5. Evaluate:
   \`python -m src.models.evaluate_model\`

## Project layout
- \`data/raw\`: original DLR data (read-only)
- \`data/interim\`: intermediate transforms
- \`data/processed\`: model-ready data
- \`src/\`: reusable pipeline + ML code
- \`notebooks/\`: exploration/prototyping
- \`experiments/\`: configs, logs, metrics
- \`models/\`: trained weights/checkpoints
- \`reports/\`: figures & writeups
- \`webapp/\`: deployment (e.g., Streamlit)

"@
Set-Content -Path "$ROOT\README.md" -Value $readme -Encoding UTF8

$gitignore = @"
# Python
.venv/
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
.ipynb_checkpoints/

# Data & artifacts
data/raw/
data/interim/
data/processed/
models/
experiments/
reports/figures/
*.ckpt
*.pt
*.pth
*.h5

# OS / IDE
.DS_Store
Thumbs.db
.vscode/
.idea/

# Logs
logs/
*.log
"@
Set-Content -Path "$ROOT\.gitignore" -Value $gitignore -Encoding UTF8

$requirements = @"
# Core
numpy
pandas
scikit-learn
matplotlib
opencv-python
tqdm
pyyaml

# Deep learning (pick one stack; both listed for flexibility)
torch
torchvision
# tensorflow

# Geospatial / imagery
rasterio
shapely

# Experiment tracking (optional)
mlflow

# Web app (optional)
streamlit
"@
Set-Content -Path "$ROOT\requirements.txt" -Value $requirements -Encoding UTF8

# --- src package init ---
Set-Content -Path "$ROOT\src\__init__.py" -Value "" -Encoding UTF8
Set-Content -Path "$ROOT\src\data_pipeline\__init__.py" -Value "" -Encoding UTF8
Set-Content -Path "$ROOT\src\models\__init__.py" -Value "" -Encoding UTF8
Set-Content -Path "$ROOT\src\utils\__init__.py" -Value "" -Encoding UTF8

# --- utils ---
$utils_common = @"
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

def ensure_dirs():
    for d in [DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, EXPERIMENTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
"@
Set-Content -Path "$ROOT\src\utils\paths.py" -Value $utils_common -Encoding UTF8

$utils_io = @"
from pathlib import Path
import json
import yaml

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
"@
Set-Content -Path "$ROOT\src\utils\io.py" -Value $utils_io -Encoding UTF8

# --- data pipeline skeleton ---
$data_prepare = @"
\"""Data preparation for DLR Aerial Crowd Dataset.

Steps:
1) Read raw imagery/labels from data/raw
2) Normalize/resize/patchify as needed
3) Export model-ready samples to data/processed
\"""
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
"@
Set-Content -Path "$ROOT\src\data_pipeline\prepare_data.py" -Value $data_prepare -Encoding UTF8

# --- model training skeleton ---
$train_model = @"
\"""Train a baseline model for event/crowd detection.

Swap this with your preferred architecture (e.g., Torch CNN, Faster R-CNN, YOLO).
\"""
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.utils.paths import PROCESSED_DIR, MODELS_DIR, ensure_dirs
import joblib
import cv2

def load_features_and_labels():
    # Placeholder: convert images to simple features (e.g., color histograms)
    X, y = [], []
    pos_dir = PROCESSED_DIR / "images"  # replace with your labeled folders
    for img_path in pos_dir.glob("*.png"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        hist = cv2.calcHist([img],[0,1,2],None,[8,8,8],[0,256,0,256,0,256]).flatten()
        X.append(hist)
        # Dummy label: replace with real labels from annotations
        y.append(0)
    return np.array(X), np.array(y)

def main():
    ensure_dirs()
    X, y = load_features_and_labels()
    if len(X) == 0:
        print("No features found. Run data prep and ensure labels are available.")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODELS_DIR / "baseline_clf.joblib")

if __name__ == "__main__":
    main()
"@
Set-Content -Path "$ROOT\src\models\train_model.py" -Value $train_model -Encoding UTF8

$eval_model = @"
\"""Evaluate trained model on held-out set or new imagery.\"""
from pathlib import Path
import joblib
from sklearn.metrics import classification_report
from src.utils.paths import MODELS_DIR

def main():
    model_path = MODELS_DIR / "baseline_clf.joblib"
    if not model_path.exists():
        print("Model not found. Train first.")
        return
    # TODO: load test features/labels and evaluate
    clf = joblib.load(model_path)
    print("Loaded model:", model_path)
    # print(classification_report(y_true, y_pred))
    print("Add your evaluation code here.")

if __name__ == "__main__":
    main()
"@
Set-Content -Path "$ROOT\src\models\evaluate_model.py" -Value $eval_model -Encoding UTF8

# --- minimal webapp (Streamlit) ---
$webapp = @"
import streamlit as st
from pathlib import Path
from PIL import Image
from src.utils.paths import PROCESSED_DIR, MODELS_DIR

st.set_page_config(page_title='DLR Aerial Crowd – Demo', layout='wide')
st.title('DLR Aerial Crowd – Event Detection Demo')

images_dir = PROCESSED_DIR / 'images'
if not images_dir.exists():
    st.warning('No processed images found. Run the data pipeline first.')
else:
    imgs = list(images_dir.glob('*.png'))[:24]
    cols = st.columns(4)
    for i, p in enumerate(imgs):
        with cols[i % 4]:
            st.image(Image.open(p), caption=p.name, use_container_width=True)

st.info('Hook your trained model here to run inference on uploaded or processed images.')
"@
Set-Content -Path "$ROOT\webapp\app.py" -Value $webapp -Encoding UTF8

# --- notebook stub ---
$nb_stub = @"
# Exploration notebook placeholder.
# Use this to inspect raw DLR files, label formats, and try quick prototypes.
"@
Set-Content -Path "$ROOT\notebooks\exploration.md" -Value $nb_stub -Encoding UTF8

Write-Host "✅ Project scaffolding created at $ROOT"
Write-Host "Next steps:"
Write-Host "1) Place and extract the DLR dataset into: $ROOT\data\raw"
Write-Host "2) Create venv and install deps: python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt"
Write-Host "3) Run pipeline: python -m src.data_pipeline.prepare_data"
Write-Host "4) Train: python -m src.models.train_model"
Write-Host "5) Launch demo: streamlit run webapp/app.py"
