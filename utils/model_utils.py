import torch
from pathlib import Path
from models.ecg_model import ECGCNN  # adjust if model file is elsewhere

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "ecg_cnn_baseline.pth"

device = torch.device("cpu")

def load_ecg_model():
    model = ECGCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

def predict_ecg_risk(model, ecg_signal):
    x = torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)[0, 1].item()
    return prob