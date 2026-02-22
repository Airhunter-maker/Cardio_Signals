"""
core/inference.py
Run model inference, SHAP explanation, and ECG saliency computation.
"""

import os
import sys
import numpy as np
import torch
import shap

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

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


def run_inference(model, scaler, feature_vector: list,
                  model_choice: str = "Random Forest") -> float:
    """
    Run a single-sample inference.
    Returns probability of positive class (CVD) as float [0,1].
    """
    import numpy as np
    X = np.array(feature_vector).reshape(1, -1)
    try:
        if model_choice == "Logistic Regression" and scaler is not None:
            X_input = scaler.transform(X)
        else:
            X_input = X
        prob = float(model.predict_proba(X_input)[0][1])
        return prob
    except Exception:
        return 0.5


def compute_shap(model, scaler, feature_vector: list,
                 model_choice: str = "Random Forest"):
    """
    Compute SHAP values for a single sample.
    Returns (shap_values_array, display_feature_names).
    """
    import numpy as np
    import pandas as pd

    X_sample = np.array(feature_vector).reshape(1, -1)

    try:
        if model_choice == "Logistic Regression" and scaler is not None:
            X_input = scaler.transform(X_sample)
            # Background data: zeros (scaled)
            background = np.zeros((1, len(FEATURES)))
            explainer = shap.LinearExplainer(
                model, background, feature_perturbation="interventional")
            shap_vals = explainer.shap_values(X_input)
        else:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_sample)
            # TreeExplainer returns list [class0, class1] for binary
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # positive class

        # Flatten to 1-D
        vals = np.array(shap_vals).flatten()
        display_names = [FEATURE_DISPLAY[f] for f in FEATURES]
        return vals, display_names
    except Exception as e:
        # Return zeros if SHAP fails
        return np.zeros(len(FEATURES)), [FEATURE_DISPLAY[f] for f in FEATURES]


def compute_saliency(model, signal: np.ndarray):
    """
    Gradient saliency for ECG CNN.
    signal shape: (500,)
    Model handles (batch, 500) internally — adds channel dim via unsqueeze.
    Returns (signal_out, saliency_norm) both shape (500,).
    """
    model.eval()
    # (1, 500) — let model's forward() add channel dim
    tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    tensor.requires_grad_(True)

    output = model(tensor)
    prob = torch.softmax(output, dim=1)[0, 1]
    model.zero_grad()
    prob.backward()

    saliency = tensor.grad.abs().squeeze().cpu().detach().numpy()
    signal_out = tensor.detach().squeeze().cpu().numpy()
    return signal_out, saliency
