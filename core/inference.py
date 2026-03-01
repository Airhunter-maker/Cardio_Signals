"""Inference, SHAP, and saliency computation."""
import numpy as np
import streamlit as st


def run_general_logic(age_years, gender, height, weight, ap_hi, ap_lo,
                      cholesterol, gluc, smoke, alco, active) -> float:
    """Manual deterministic scoring based on clinical heuristics."""
    try:
        from core.validators import level_to_int, calculate_bmi

        points = 0

        # Age
        if age_years > 60:
            points += 3
        elif age_years > 50:
            points += 2
        elif age_years > 40:
            points += 1

        # Blood Pressure using proper AHA thresholds
        try:
            ap_hi_int = int(ap_hi)
            ap_lo_int = int(ap_lo)
        except (TypeError, ValueError):
            ap_hi_int, ap_lo_int = 120, 80

        if ap_hi_int > 180 or ap_lo_int > 120:
            points += 5  # Hypertensive crisis
        elif ap_hi_int >= 140 or ap_lo_int >= 90:
            points += 4  # Stage 2
        elif 130 <= ap_hi_int <= 139 or 80 <= ap_lo_int <= 89:
            points += 2  # Stage 1
        elif 120 <= ap_hi_int <= 129 and ap_lo_int < 80:
            points += 1  # Elevated

        # Cholesterol & Glucose - safely converted from text to numeric
        chol_val = level_to_int(cholesterol, default=1)
        gluc_val = level_to_int(gluc, default=1)
        points += (chol_val - 1)  # 0, 1, or 2 extra points
        points += (gluc_val - 1)

        # BMI
        bmi = calculate_bmi(height, weight)
        if bmi >= 30:
            points += 2
        elif bmi >= 25:
            points += 1

        # Lifestyle
        if smoke:
            points += 2
        if alco:
            points += 1
        if not active:
            points += 1

        # Gender risk adjustment (males have slightly higher baseline risk)
        if gender == "Male":
            points += 1

        # Scale points (max ~18) to 0-1 probability
        base_prob = 0.05
        prob = base_prob + (points / 18.0) * 0.9
        return float(min(1.0, max(0.0, prob)))

    except Exception as e:
        # Log internally, return neutral 50% rather than crashing
        if "errors" in st.session_state:
            st.session_state["errors"].append(f"Scoring error: {e}")
        return 0.5


FEATURE_NAMES = ["age", "gender", "height", "weight",
                 "ap_hi", "ap_lo", "cholesterol", "gluc",
                 "smoke", "alco", "active"]

FEATURE_LABELS = {
    "age":         "Age",
    "gender":      "Gender",
    "height":      "Height",
    "weight":      "Weight",
    "ap_hi":       "Systolic BP",
    "ap_lo":       "Diastolic BP",
    "cholesterol": "Cholesterol",
    "gluc":        "Glucose",
    "smoke":       "Smoker",
    "alco":        "Alcohol Use",
    "active":      "Physically Active",
}


def build_feature_vector(age_years: int, gender: str,
                         height: int, weight: float,
                         ap_hi: int, ap_lo: int,
                         cholesterol: str, gluc: str,
                         smoke: bool, alco: bool,
                         active: bool) -> np.ndarray:
    """Build feature vector matching training column order."""
    try:
        from core.validators import level_to_int

        chol_map = {"Female": 1, "Male": 2}
        gender_val = chol_map.get(str(gender), 1)

        # Safely convert cholesterol/gluc from text to int
        chol_int = level_to_int(cholesterol, default=1)
        gluc_int = level_to_int(gluc, default=1)

        vec = np.array([
            int(age_years) * 365,   # age in days (training format)
            gender_val,
            int(height),
            float(weight),
            int(ap_hi),
            int(ap_lo),
            chol_int,
            gluc_int,
            int(bool(smoke)),
            int(bool(alco)),
            int(bool(active)),
        ], dtype=float).reshape(1, -1)
        return vec

    except Exception as e:
        if "errors" in st.session_state:
            st.session_state["errors"].append(f"Feature vector error: {e}")
        return np.zeros((1, 11), dtype=float)


def run_inference(features: np.ndarray, models: dict,
                  model_choice: str) -> float:
    """Run clinical model inference. Returns probability 0-1."""
    try:
        if features is None or features.shape[-1] != 11:
            raise ValueError("Invalid feature vector shape")

        if model_choice == "Random Forest":
            if models.get("rf") is None:
                raise ValueError("Random Forest model not loaded")
            prob = models["rf"].predict_proba(features)[0][1]
        else:
            if models.get("lr") is None or models.get("scaler") is None:
                raise ValueError("Logistic Regression model not loaded")
            scaled = models["scaler"].transform(features)
            prob = models["lr"].predict_proba(scaled)[0][1]

        return float(np.clip(prob, 0.0, 1.0))

    except Exception as e:
        if "errors" in st.session_state:
            st.session_state["errors"].append(f"Inference error: {e}")
        return 0.5


def compute_shap(features: np.ndarray, models: dict,
                 model_choice: str):
    """Compute SHAP values for the given feature vector."""
    try:
        import shap
        import pandas as pd
        import os

        def get_background():
            paths = ["cardio_base.csv", "data/raw/cardio_base.csv",
                     "data/cardio_base.csv"]
            for p in paths:
                if os.path.exists(p):
                    df = pd.read_csv(p, sep=";")
                    df = df[(df["ap_hi"] >= 70) & (df["ap_hi"] <= 250)]
                    df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 150)]
                    return df[FEATURE_NAMES].sample(
                        min(500, len(df)), random_state=42).values
            return np.zeros((1, 11))

        if model_choice == "Random Forest":
            explainer = shap.TreeExplainer(models["rf"])
            shap_vals = explainer.shap_values(features)
            if isinstance(shap_vals, list):
                return shap_vals[1][0]
            return shap_vals[0]
        else:
            bg = get_background()
            bg_scaled = models["scaler"].transform(bg)
            feat_scaled = models["scaler"].transform(features)
            explainer = shap.LinearExplainer(
                models["lr"], bg_scaled,
                feature_perturbation="interventional")
            shap_vals = explainer.shap_values(feat_scaled)
            return shap_vals[0]
    except Exception as e:
        if "errors" in st.session_state:
            st.session_state["errors"].append(f"SHAP error: {e}")
        return np.zeros(11)


def compute_saliency(model, signal: np.ndarray):
    """
    Gradient saliency for ECGCNN.
    signal: 1D numpy array of length 500
    """
    try:
        import torch
        tensor = torch.tensor(
            signal, dtype=torch.float32).unsqueeze(0)
        tensor.requires_grad_(True)
        output = model(tensor)
        prob = torch.softmax(output, dim=1)[0, 1]
        model.zero_grad()
        prob.backward()
        saliency = tensor.grad.abs().squeeze().detach().numpy()
        signal_out = tensor.detach().squeeze().numpy()
        return signal_out, saliency, prob.item()
    except Exception as e:
        if "errors" in st.session_state:
            st.session_state["errors"].append(f"Saliency error: {e}")
        arr = signal if isinstance(signal, np.ndarray) else np.zeros(500)
        return arr, np.zeros(len(arr)), 0.5