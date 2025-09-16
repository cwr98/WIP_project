# DLR Aerial Crowd Event Detection

This repo contains a geospatial ML pipeline using the **DLR Aerial Crowd Dataset**.

## Quick start
1. Put the original dataset archive(s) in data/raw/ and extract them **there**.
2. Create a virtual environment:
   \python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt\
3. Run the data prep:
   \python -m src.data_pipeline.prepare_data\
4. Train:
   \python -m src.models.train_model\
5. Evaluate:
   \python -m src.models.evaluate_model\

## Project layout
- \data/raw\: original DLR data (read-only)
- \data/interim\: intermediate transforms
- \data/processed\: model-ready data
- \src/\: reusable pipeline + ML code
- \
Notebooks/\: exploration/prototyping
- \experiments/\: configs, logs, metrics
- \models/\: trained weights/checkpoints
- \
Reports/\: figures & writeups
- \webapp/\: deployment (e.g., Streamlit)

