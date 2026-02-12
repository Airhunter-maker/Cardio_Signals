import numpy as np
from PIL import Image

def load_ecg_image(uploaded_file):
    img = Image.open(uploaded_file).convert("L")
    img = img.resize((500, 1))

    ecg_signal = np.array(img).flatten().astype(float)
    ecg_signal = (ecg_signal - ecg_signal.mean()) / (ecg_signal.std() + 1e-8)

    return ecg_signal