"""
core/model_loader.py
Loading all CardioSignals models. Retrains sklearn models on-the-fly if .pkl files are missing.
"""

import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import torch

# ── Add repo root to path so models/ecg_model.py is importable ──
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.ecg_model import ECGCNN


FEATURES = ['age', 'gender', 'height', 'weight',
            'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
            'smoke', 'alco', 'active']


@st.cache_resource(show_spinner=False)
def load_clinical_models():
    """Load or retrain Logistic Regression and Random Forest models."""
    lr_path  = os.path.join(_ROOT, 'models', 'log_reg_model.pkl')
    rf_path  = os.path.join(_ROOT, 'models', 'random_forest_model.pkl')
    sc_path  = os.path.join(_ROOT, 'models', 'scaler.pkl')

    if os.path.exists(lr_path) and os.path.exists(rf_path) and os.path.exists(sc_path):
        try:
            lr = joblib.load(lr_path)
            rf = joblib.load(rf_path)
            sc = joblib.load(sc_path)
            return lr, rf, sc, True
        except Exception as e:
            st.session_state["errors"].append(f"Model load error: {e}")

    # ── CASE B: Retrain ──
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        # Try repo root first, then data/raw as fallback
        csv_path = os.path.join(_ROOT, 'cardio_base.csv')
        if not os.path.exists(csv_path):
            csv_path = os.path.join(_ROOT, 'data', 'raw', 'cardio_base.csv')

        df = pd.read_csv(csv_path, sep=';')
        df = df[(df['ap_hi'] >= 70) & (df['ap_hi'] <= 250)]
        df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 150)]

        X = df[FEATURES]
        y = df['cardio']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train_scaled, y_train)

        rf = RandomForestClassifier(n_estimators=200,
             random_state=42, class_weight='balanced')
        rf.fit(X_train, y_train)

        os.makedirs(os.path.join(_ROOT, 'models'), exist_ok=True)
        joblib.dump(lr, lr_path)
        joblib.dump(rf, rf_path)
        joblib.dump(scaler, sc_path)

        return lr, rf, scaler, True

    except Exception as e:
        st.session_state["errors"].append(f"Retrain error: {e}")
        return None, None, None, False


@st.cache_resource(show_spinner=False)
def load_ecg_model():
    """Load pre-trained ECGCNN from .pth file."""
    pth_path = os.path.join(_ROOT, 'models', 'ecg_cnn_baseline.pth')
    try:
        model = ECGCNN()
        model.load_state_dict(
            torch.load(pth_path, map_location='cpu'))
        model.eval()
        return model, True
    except Exception as e:
        st.session_state["errors"].append(f"ECG model load error: {e}")
        return None, False


def load_all_models():
    """Convenience wrapper. Returns dict with all models."""
    lr, rf, scaler, clinical_ok = load_clinical_models()
    ecg_model, ecg_ok = load_ecg_model()

    st.session_state["clinical_model_loaded"] = clinical_ok
    st.session_state["ecg_model_loaded"] = ecg_ok

    return {
        "lr":      lr,
        "rf":      rf,
        "scaler":  scaler,
        "ecg":     ecg_model,
        "clinical_ok": clinical_ok,
        "ecg_ok":  ecg_ok,
    }
