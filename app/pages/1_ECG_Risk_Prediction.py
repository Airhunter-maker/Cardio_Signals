import streamlit as st
import matplotlib.pyplot as plt
from utils.ecg_utils import load_ecg_image
from utils.model_utils import load_ecg_model, predict_ecg_risk

st.title("📈 ECG-Based Heart Risk Prediction")

st.markdown("""
This page demonstrates **early cardiovascular risk detection using ECG signals only**.
""")

uploaded = st.file_uploader("Upload an ECG image", type=["png", "jpg", "jpeg"])

model = load_ecg_model()

if uploaded:
    ecg_signal = load_ecg_image(uploaded)
    risk = predict_ecg_risk(model, ecg_signal)

    st.metric("Predicted Heart Risk Probability", f"{risk:.2f}")

    fig, ax = plt.subplots()
    ax.plot(ecg_signal)
    ax.set_title("Processed ECG Signal")
    st.pyplot(fig)

    st.markdown("""
### Interpretation
Higher scores indicate ECG patterns associated with increased cardiovascular risk.
""")