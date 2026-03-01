# CardioSignals 

**AI-Powered Cardiovascular Risk Intelligence**

CardioSignals is a web application that predicts cardiovascular disease risk using machine learning. It analyzes 11 clinical features (age, blood pressure, cholesterol, lifestyle) to provide instant risk assessments with explainable AI insights. The platform combines Random Forest classification with a 1D CNN for ECG signal analysis, helping identify at-risk patients through transparent, interpretable predictions.

This is an educational and research tool, not a medical diagnostic device.

**Project Link**: 

---

## 📌 Problem Statement

Cardiovascular disease (CVD) is the leading cause of death globally, accounting for **17.9 million deaths annually**. Despite this, early detection remains challenging:

- **1 in 4 deaths** in developed countries is caused by heart disease
- Traditional risk assessment (Framingham, SCORE) requires manual calculation and expertise
- ECG analysis demands specialized cardiological training and is time-consuming
- Many at-risk individuals remain undiagnosed until a major cardiac event
- Existing ML models lack transparency, reducing trust among clinicians

**Key Gaps:**
- ⚠️ Limited access to automated risk screening tools
- ⚠️ No real-time, explainable AI for cardiovascular assessment
- ⚠️ Black-box models prevent clinical adoption

---

## 🎯 Project Objectives

- **Automate cardiovascular risk assessment** using machine learning (target: ≥75% accuracy)
- **Provide explainable predictions** through SHAP (SHapley Additive exPlanations)
- **Analyze ECG signals** using 1D CNN with gradient-based saliency mapping
- **Create an accessible interface** for healthcare professionals and researchers
- **Demonstrate transparent AI** that augments (not replaces) clinical judgment

---

## 🧩 System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Streamlit)                │
│  🏠 Home  |  🔍 Risk Analyzer  |  📈 ECG Signal Explorer    │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────┴─────────────────────────────────────┐
│                   APPLICATION LOGIC LAYER                     │
│  • Input Validation  • Feature Engineering  • Routing        │
└────────────────────────┬─────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         ↓                               ↓
┌─────────────────────────┐   ┌──────────────────────────┐
│  CLINICAL ML PIPELINE   │   │  ECG DL PIPELINE         │
│                         │   │                          │
│  • Random Forest (100)  │   │  • 1D CNN (4.1M params) │
│  • SHAP Explainer       │   │  • Saliency Mapping      │
│  • Risk Score (0-100%)  │   │  • Risk Classification   │
└─────────────────────────┘   └──────────────────────────┘
```

**Data Flow:**
```
User Input → Validation → Feature Engineering → Model Prediction → SHAP Analysis → Results Display
```

**Tech Stack:**
- **Frontend**: Streamlit (Python-native web framework)
- **ML**: scikit-learn (Random Forest), TensorFlow (1D CNN)
- **Explainability**: SHAP library for feature importance
- **Visualization**: Plotly for interactive charts
- **Deployment**: Streamlit Cloud

---

## 🚀 Features

### 🏠 Home Page
- Animated ECG hero section with call-to-action
- Feature highlights (Real-Time Risk, Explainable AI, ECG Analysis)
- "How It Works" 3-step guide
- Responsive design for desktop/mobile

### 🔍 Risk Analyzer (Primary Feature)
**Input Form:**
- **Demographics**: Age, Gender, Height, Weight (with live BMI calculation)
- **Vitals**: Systolic/Diastolic BP (with real-time categorization: Normal/Stage 1/Stage 2)
- **Labs**: Cholesterol, Glucose (Normal/Above Normal/High)
- **Lifestyle**: Smoking, Alcohol, Physical Activity

**Results:**
- **Risk Gauge**: Visual gauge showing 0-100% risk score
- **Risk Badge**: Color-coded (🟢 LOW / 🟡 MODERATE / 🔴 HIGH)
- **Top 3 Contributing Factors**: SHAP-based importance ranking
```
  Example:
  1. Age (45 years)        ████████████░░░░  +0.25
  2. Systolic BP (135)     ██████████░░░░░░  +0.18
  3. Weight (85 kg)        ███████░░░░░░░░░  +0.12
```
- **Clinical Interpretation**: Plain English explanation + recommendations

### 📈 ECG Signal Explorer
- **Waveform Viewer**: Interactive ECG signal plot (100 samples)
- **Saliency Mapping**: Toggle "Show Risk Regions" to highlight abnormalities
- **Risk Scoring**: CNN-based risk classification for each signal
- **Educational**: "Our AI analyzes heart signal patterns to detect risk markers"

### 🧠 Explainability
- **SHAP Values**: Quantifies each feature's contribution to prediction
- **Gradient Saliency**: Visualizes which ECG segments drive CNN predictions
- **Transparent**: Users see exactly why model made its prediction

---

## 🛠️ Tech Stack

**Frontend**
- Streamlit 1.28+ (Web framework)
- Plotly 5.17+ (Interactive charts)
- Custom CSS/HTML for styling

**Backend / ML**
- Python 
- scikit-learn 1.3+ (Random Forest, preprocessing)
- TensorFlow 2.13+ (1D CNN for ECG)
- SHAP 0.42+ (Explainability)
- Pandas, NumPy (Data handling)

**Models**
- **Random Forest Classifier**: 100 trees, 11 features, 73.2% accuracy, 0.784 AUC, trained on 68,754 ECG signals

**Deployment**
- Streamlit Cloud

---

## ⚠️ Ethical Disclaimer

**CardioSignals is NOT a medical device and should NOT be used for clinical diagnosis.**

This application is designed for:
- ✅ Research and educational purposes
- ✅ Demonstrating explainable AI in healthcare
- ✅ Preliminary risk screening for further study

This application is **NOT** suitable for:
- Clinical diagnosis or treatment decisions
- Replacing professional medical judgment
- Emergency cardiac situations

**Always consult qualified healthcare professionals for medical decisions.**

---


**Explainable AI:**
- SHAP provides consistent, theoretically sound feature importance
- Visualizations (gauges, bar charts) more effective than raw numbers for user trust
- Real-time feedback (BP categorization, BMI) improves input accuracy

---

## 🔮 Future Enhancements

**Model Improvements:**
- Incorporate family history, medications, lab values (LDL, HbA1c)
- Multi-task learning: predict event type (MI, stroke), time-to-event (survival analysis)
- Transfer learning: pre-train ECG CNN on large public datasets (PTB-XL, MIMIC-III)
- Uncertainty quantification: show confidence intervals (e.g., "35% ± 7%")

---

## 📄 License

This project is developed for **academic and educational purposes**.
