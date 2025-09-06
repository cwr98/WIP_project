import streamlit as st
from pathlib import Path
from PIL import Image
from src.utils.paths import PROCESSED_DIR, MODELS_DIR

st.set_page_config(page_title='DLR Aerial Crowd â€“ Demo', layout='wide')
st.title('DLR Aerial Crowd â€“ Event Detection Demo')

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
