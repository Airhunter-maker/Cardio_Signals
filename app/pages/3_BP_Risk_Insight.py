import streamlit as st

st.title("🩺 Blood Pressure & ECG")

st.markdown("""
Hypertension affects cardiac electrical activity.

### ECG correlates of high BP:
- Increased QRS amplitude
- Ventricular hypertrophy patterns
- ST-segment changes
""")

st.info("Our ECG model captures these indirect patterns without measuring BP explicitly.")