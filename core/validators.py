"""Input validation and categorization for CardioSignals."""
from typing import Tuple, Optional


# ── BLOOD PRESSURE CATEGORIZATION ─────────────────────────────────────
def categorize_bp(systolic: int, diastolic: int) -> Tuple[str, str]:
    """
    Categorize blood pressure using AHA guidelines.
    Returns (category_label, hex_color).
    """
    try:
        systolic = int(systolic)
        diastolic = int(diastolic)

        if systolic > 180 or diastolic > 120:
            return "Hypertensive Crisis", "#991B1B"
        elif systolic >= 140 or diastolic >= 90:
            return "High Blood Pressure (Stage 2)", "#DC2626"
        elif 130 <= systolic <= 139 or 80 <= diastolic <= 89:
            return "High Blood Pressure (Stage 1)", "#EF4444"
        elif 120 <= systolic <= 129 and diastolic < 80:
            return "Elevated", "#F59E0B"
        elif systolic < 120 and diastolic < 80:
            return "Normal", "#10B981"
        else:
            return "Normal", "#10B981"
    except (TypeError, ValueError):
        return "Unknown", "#64748B"


def bp_color(systolic: int, diastolic: int) -> str:
    """Return just the color for BP categorization."""
    _, color = categorize_bp(systolic, diastolic)
    return color


# ── CHOLESTEROL / GLUCOSE MAPPING ──────────────────────────────────────
LEVEL_MAP = {
    # New user-facing labels
    "Low":              1,
    "Normal":           2,
    "High":             3,
    # Legacy labels (kept as fallback)
    "Above Normal":     2,
    "Well Above Normal":3,
}

def level_to_int(level: str, default: int = 1) -> int:
    """Map text level selection to numeric value (1/2/3)."""
    try:
        return LEVEL_MAP.get(str(level).strip(), default)
    except Exception:
        return default


# ── NUMERIC VALIDATION ─────────────────────────────────────────────────
def validate_inputs(
    age_years: int,
    height: int,
    weight: float,
    ap_hi: int,
    ap_lo: int,
) -> Tuple[bool, Optional[str]]:
    """
    Validate all numeric inputs are within reasonable clinical ranges.
    Returns (is_valid, error_message_or_None).
    """
    errors = []

    if not (18 <= age_years <= 120):
        errors.append("Age must be between 18 and 120 years.")
    if not (100 <= height <= 250):
        errors.append("Height must be between 100 and 250 cm.")
    if not (20 <= weight <= 300):
        errors.append("Weight must be between 20 and 300 kg.")
    if not (60 <= ap_hi <= 300):
        errors.append("Systolic BP must be between 60 and 300 mmHg.")
    if not (30 <= ap_lo <= 200):
        errors.append("Diastolic BP must be between 30 and 200 mmHg.")
    if ap_lo >= ap_hi:
        errors.append(
            "Diastolic BP must be lower than Systolic BP. "
            "Please check your blood pressure values."
        )

    if errors:
        return False, " | ".join(errors)
    return True, None


# ── BMI CALCULATION ─────────────────────────────────────────────────────
def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    """Calculate BMI from height in cm and weight in kg."""
    try:
        h = float(height_cm) / 100.0
        if h <= 0:
            return 0.0
        return float(weight_kg) / (h ** 2)
    except Exception:
        return 0.0


def categorize_bmi(bmi: float) -> Tuple[str, str]:
    """Return (bmi_category, color) for the given BMI value."""
    if bmi < 18.5:
        return "Underweight", "#7C3AED"
    elif bmi < 25:
        return "Normal Weight", "#10B981"
    elif bmi < 30:
        return "Overweight", "#F59E0B"
    else:
        return "Obese", "#EF4444"
