"""Data loading utilities."""
import os
import streamlit as st
import pandas as pd
import numpy as np


@st.cache_data(show_spinner=False)
def load_cardio_data() -> pd.DataFrame:
    """Load and clean cardio_base.csv."""
    paths = ["cardio_base.csv", "data/raw/cardio_base.csv",
             "data/cardio_base.csv"]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p, sep=";")
            df = df[(df["ap_hi"] >= 70) & (df["ap_hi"] <= 250)]
            df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 150)]
            df["age_years"] = df["age"] // 365
            return df
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_ecg_data(n_samples: int = 100) -> np.ndarray:
    """Load normalized ECG array, first n_samples."""
    paths = ["data/processed/X_ecg_normalized.npy",
             "X_ecg_normalized.npy"]
    for p in paths:
        if os.path.exists(p):
            arr = np.load(p, allow_pickle=True)
            return arr[:n_samples].astype(np.float32)
    return None


@st.cache_data(show_spinner=False)
def load_ecg_segments():
    """Load ECG segments and labels."""
    seg_paths = ["data/processed/X_ecg_segments.npy",
                 "X_ecg_segments.npy"]
    lbl_paths = ["data/processed/y_ecg_labels.npy",
                 "y_ecg_labels.npy"]
    X, y = None, None
    for p in seg_paths:
        if os.path.exists(p):
            X = np.load(p, allow_pickle=True).astype(np.float32)
            break
    for p in lbl_paths:
        if os.path.exists(p):
            y = np.load(p, allow_pickle=True)
            break
    return X, y


@st.cache_data(show_spinner=False)
def load_metrics_csv(filename: str) -> pd.DataFrame:
    """Load a CSV from results/metrics/."""
    paths = [
        f"results/metrics/{filename}",
        f"results/{filename}",
        filename,
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return pd.DataFrame()