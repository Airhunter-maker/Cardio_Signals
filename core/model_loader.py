"""Model loading with auto-retrain fallback."""
import os
import streamlit as st
import pandas as pd
import numpy as np


@st.cache_resource(show_spinner=False)
def load_all_models() -> dict:
    """
    Load all models. If sklearn pkl files missing, retrain and save.
    Returns dict with keys: rf, lr, scaler, ecg
    """
    import joblib

    result = {"rf": None, "lr": None, "scaler": None, "ecg": None}

    # ── SKLEARN MODELS ────────────────────────────────────────────
    rf_path  = "models/random_forest_model.pkl"
    lr_path  = "models/log_reg_model.pkl"
    scaler_path = "models/scaler.pkl"

    if (os.path.exists(rf_path) and
            os.path.exists(lr_path) and
            os.path.exists(scaler_path)):
        try:
            result["rf"]     = joblib.load(rf_path)
            result["lr"]     = joblib.load(lr_path)
            result["scaler"] = joblib.load(scaler_path)
        except Exception as e:
            st.session_state["errors"].append(
                f"Model load error: {e}")
    else:
        # Retrain from data
        try:
            result["rf"], result["lr"], result["scaler"] = _retrain()
        except Exception as e:
            st.session_state["errors"].append(
                f"Retrain error: {e}")

    # ── ECG CNN ───────────────────────────────────────────────────
    pth_path = "models/ecg_cnn_baseline.pth"
    if os.path.exists(pth_path):
        try:
            import torch
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                            .replace("/core", "").replace("\\core", ""))
            from models.ecg_model import ECGCNN
            model = ECGCNN()
            model.load_state_dict(
                torch.load(pth_path, map_location="cpu"))
            model.eval()
            result["ecg"] = model
        except Exception as e:
            st.session_state["errors"].append(
                f"ECG model load error: {e}")

    return result


def _retrain():
    """Retrain sklearn models from raw data and save."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    # Find data file
    data_paths = [
        "cardio_base.csv",
        "data/raw/cardio_base.csv",
        "data/cardio_base.csv",
    ]
    df = None
    for p in data_paths:
        if os.path.exists(p):
            df = pd.read_csv(p, sep=";")
            break
    if df is None:
        raise FileNotFoundError(
            "cardio_base.csv not found. Place it in the project root "
            "or data/raw/")

    # Clean
    df = df[(df["ap_hi"] >= 70) & (df["ap_hi"] <= 250)]
    df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 150)]

    FEATURES = ["age", "gender", "height", "weight",
                "ap_hi", "ap_lo", "cholesterol", "gluc",
                "smoke", "alco", "active"]
    X = df[FEATURES]
    y = df["cardio"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(rf,     "models/random_forest_model.pkl")
    joblib.dump(lr,     "models/log_reg_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    return rf, lr, scaler