"""
core/data_loader.py
Load clinical data, ECG arrays, and build feature vectors for inference.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURES = ['age', 'gender', 'height', 'weight',
            'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
            'smoke', 'alco', 'active']

FEATURE_DISPLAY = {
    'age':         'Age',
    'gender':      'Gender',
    'height':      'Height',
    'weight':      'Weight',
    'ap_hi':       'Systolic BP',
    'ap_lo':       'Diastolic BP',
    'cholesterol': 'Cholesterol',
    'gluc':        'Glucose',
    'smoke':       'Smoker',
    'alco':        'Alcohol Use',
    'active':      'Physically Active',
}


@st.cache_data(show_spinner=False)
def load_cardio_data() -> pd.DataFrame | None:
    """Load and clean cardio_base.csv."""
    paths = [
        os.path.join(_ROOT, 'cardio_base.csv'),
        os.path.join(_ROOT, 'data', 'raw', 'cardio_base.csv'),
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, sep=';')
                df = df[(df['ap_hi'] >= 70) & (df['ap_hi'] <= 250)]
                df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 150)]
                # Convert age from days to years
                if df['age'].median() > 365:
                    df['age_years'] = (df['age'] / 365).round(1)
                else:
                    df['age_years'] = df['age']
                return df
            except Exception as e:
                st.session_state.setdefault("errors", []).append(
                    f"CSV load error: {e}")
    return None


@st.cache_data(show_spinner=False)
def load_ecg_data():
    """Load ECG numpy arrays. Returns (X_norm, y_labels, X_segs) or (None, None, None)."""
    proc = os.path.join(_ROOT, 'data', 'processed')
    try:
        X_norm = np.load(os.path.join(proc, 'X_ecg_normalized.npy'))
        y = np.load(os.path.join(proc, 'y_ecg_labels.npy'))
        X_segs = np.load(os.path.join(proc, 'X_ecg_segments.npy'))
        return X_norm, y, X_segs
    except Exception as e:
        st.session_state.setdefault("errors", []).append(
            f"ECG array load error: {e}")
        return None, None, None


def preprocess_input(age_y, gender_str, height, weight,
                     ap_hi, ap_lo, chol_str, gluc_str,
                     smoke, alco, active) -> list:
    """
    Convert human-readable form inputs to model feature vector.
    age in YEARS is converted to days for the model.
    Returns list matching FEATURES order.
    """
    chol_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
    gender_map = {"Female": 1, "Male": 2}

    return [
        age_y * 365,                    # age → days
        gender_map[gender_str],
        height,
        weight,
        ap_hi,
        ap_lo,
        chol_map[chol_str],
        chol_map[gluc_str],
        int(smoke),
        int(alco),
        int(active),
    ]


def get_feature_display_names(features: list) -> list:
    """Map raw feature names to human-readable display names."""
    return [FEATURE_DISPLAY.get(f, f) for f in features]
