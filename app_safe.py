
import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Disease Detection Prediction (Safe)", layout="centered")

st.title("Disease Detection Prediction — Safe Mode")
st.caption("Returns **Insufficient data** for invalid/out-of-range/low-confidence inputs to avoid misdiagnosis.")

# ---------- Load model & encoder ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.joblib")
    try:
        le = joblib.load("label_encoder.joblib")
    except Exception:
        le = None
    return model, le

try:
    best_model, le = load_artifacts()
    st.success("Model (and label encoder if present) loaded successfully.")
except FileNotFoundError:
    st.error("Model or LabelEncoder not found. Ensure 'best_model.joblib' and 'label_encoder.joblib' are in the same directory.")
    st.stop()

# ---------- Feature schema & ranges ----------
# Keep the exact same features the original app expected
feature_names = [
    'Memory Recall (%)', 'Gait Speed (m/s)', 'Tremor Frequency (Hz)', 'Speech Rate (wpm)',
    'Reaction Time (ms)', 'Eye Movement Irregularities (saccades/s)', 'Sleep Disturbance (scale 0-10)',
    'Cognitive Test Score (MMSE)', 'Blood Pressure (mmHg)', 'Cholesterol (mg/dL)',
    'Diabetes', 'Severity', 'Gender_Male'
]

# Conservative plausibility ranges (adjustable via sidebar if needed)
default_ranges = {
    'Memory Recall (%)': (0, 100),
    'Gait Speed (m/s)': (0.0, 3.0),
    'Tremor Frequency (Hz)': (0.0, 20.0),
    'Speech Rate (wpm)': (0.0, 300.0),
    'Reaction Time (ms)': (50.0, 2000.0),
    'Eye Movement Irregularities (saccades/s)': (0.0, 10.0),
    'Sleep Disturbance (scale 0-10)': (0.0, 10.0),
    'Cognitive Test Score (MMSE)': (0.0, 30.0),
    'Blood Pressure (mmHg)': (50.0, 250.0),
    'Cholesterol (mg/dL)': (50.0, 400.0),
    'Diabetes': (0, 1),
    'Severity': (0, 4),
    'Gender_Male': (0, 1),
}

st.sidebar.header("Safety Settings")
zero_ratio_max = st.sidebar.slider("Max allowed zero ratio", 0.0, 1.0, 0.20, 0.05,
                                   help="Reject if more than this fraction of inputs are exactly zero.")
use_confidence = st.sidebar.checkbox("Use probability threshold (if supported)", True)
prob_threshold = st.sidebar.slider("Min probability threshold", 0.50, 0.95, 0.70, 0.01)
# Allow user to tweak ranges
st.sidebar.subheader("Plausibility Ranges (editable)")
ranges = {}
for fname in feature_names:
    lo, hi = default_ranges[fname]
    # Create compact inputs
    cols = st.sidebar.columns([2,2])
    with cols[0]:
        lo_val = st.number_input(f"{fname} (min)", value=float(lo), key=f"{fname}_min")
    with cols[1]:
        hi_val = st.number_input(f"{fname} (max)", value=float(hi), key=f"{fname}_max")
    ranges[fname] = (float(lo_val), float(hi_val))

# ---------- Input section ----------
st.subheader("Enter Input Data")
mode = st.radio("Input mode:", ["Manual", "CSV (single row)"], horizontal=True)

def validate_vector(vec: np.ndarray):
    # Shape & numeric checks
    if vec.ndim != 1 or len(vec) != len(feature_names):
        return False, "Invalid input shape."
    if np.isnan(vec).any() or np.isinf(vec).any():
        return False, "Invalid inputs (NaN/Inf found)."
    # Zero ratio
    zero_ratio = float(np.mean(vec == 0))
    if zero_ratio > zero_ratio_max:
        return False, f"Insufficient data — too many zeros ({zero_ratio:.0%})."
    # Range checks
    oob_idx = []
    for i, fname in enumerate(feature_names):
        lo, hi = ranges[fname]
        if vec[i] < lo or vec[i] > hi:
            oob_idx.append(fname)
    if oob_idx:
        preview = ", ".join(oob_idx[:4]) + ("..." if len(oob_idx) > 4 else "")
        return False, f"Out-of-range features: {preview}."
    return True, None

def predict_with_safety(input_df: pd.DataFrame):
    x = input_df.values[0].astype(float)
    ok, reason = validate_vector(x)
    if not ok:
        return "Insufficient data", reason, None, None

    # Try confidence gate (if model supports predict_proba)
    if use_confidence and hasattr(best_model, "predict_proba"):
        try:
            proba = best_model.predict_proba(input_df)[0]
            max_p = float(np.max(proba))
            if max_p < prob_threshold:
                return "Insufficient data", f"Low confidence (p_max={max_p:.2f} < {prob_threshold:.2f}).", proba, None
        except Exception:
            proba = None
    else:
        proba = None

    try:
        pred = best_model.predict(input_df)
        if le is not None:
            try:
                pred = le.inverse_transform(pred)
            except Exception:
                pass
        label = str(pred[0])
        return label, "ok", proba, None
    except Exception as e:
        return "Insufficient data", f"Prediction error: {e}", None, None
if st.button("Predict"):
    # Assuming 'best_model' and 'le' (label encoder) are already loaded
    
    # Predict the class (encoded label)
    prediction_encoded = best_model.predict(input_df)
    
    # Get the class probabilities for each class
    prediction_proba = best_model.predict_proba(input_df)  # Returns probabilities for each class

    # Decode the predicted class (convert encoded prediction back to original labels)
    prediction_decoded = le.inverse_transform(prediction_encoded)

    # Display the predicted class
    st.write(f"The predicted disease is: **{prediction_decoded[0]}**")

    # Display the class probabilities (for each class)
    # Create a DataFrame for better display of class probabilities
    prob_df = pd.DataFrame(prediction_proba, columns=le.classes_)  # This will show probabilities for each class

    # Display the probabilities in a table format
    st.write("Class Probabilities:")
    st.dataframe(prob_df)

# Collect input
if mode == "Manual":
    cols = st.columns(min(4, len(feature_names)) or 1)
    feat_vals = {}
    for i, fname in enumerate(feature_names):
        with cols[i % len(cols)]:
            # numeric inputs; integers for binary/int-like fields
            if fname in ['Diabetes', 'Severity', 'Gender_Male', 'Sleep Disturbance (scale 0-10)']:
                feat_vals[fname] = st.number_input(fname, value=0, step=1, format="%d")
            else:
                feat_vals[fname] = st.number_input(fname, value=0.0, step=0.1, format="%.3f")
    input_df = pd.DataFrame([feat_vals])[feature_names]
else:
    st.info("Upload a CSV with exactly one row and the same column names (and order) as the feature list.")
    csv_file = st.file_uploader("Upload CSV", type=["csv"])
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        st.dataframe(df.head())
        # Simple schema enforcement
        missing = [c for c in feature_names if c not in df.columns]
        extra = [c for c in df.columns if c not in feature_names]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()
        if extra:
            st.warning(f"Extra columns in CSV ignored: {extra}")
        if df.shape[0] != 1:
            st.error("CSV must contain exactly one row.")
            st.stop()
        input_df = df[feature_names].iloc[[0]]
    else:
        input_df = pd.DataFrame(columns=feature_names)

st.subheader("Input Preview")
st.write(input_df if not input_df.empty else "No input yet.")

# ---------- Predict button ----------
if st.button("Predict"):
    if input_df.empty or input_df.shape[0] != 1:
        st.warning("Please provide exactly one row of inputs.")
    else:
        label, reason, proba, _ = predict_with_safety(input_df)
        if label == "Insufficient data":
            st.warning(f"Result: **Insufficient data** — {reason}")
        else:
            st.success(f"Prediction: **{label}**")
            if proba is not None:
                # Show class probabilities if encoder available and classes align
                try:
                    if hasattr(best_model, "classes_"):
                        classes = best_model.classes_
                        if le is not None:
                            try:
                                classes = le.inverse_transform(classes)
                            except Exception:
                                pass
                        prob_df = pd.DataFrame([proba], columns=classes)
                        st.write("Class probabilities:")
                        st.dataframe(prob_df.style.format("{:.2f}"))
                except Exception:
                    pass

st.divider()
with st.expander("Notes"):
    st.markdown(
        "- The app rejects clearly invalid inputs (NaN/Inf), too many zeros, and values outside editable plausibility ranges.\n"
        "- If your model exposes `predict_proba`, a probability threshold is applied. Lower the threshold if coverage is too small.\n"
        "- Adjust ranges in the sidebar to match your dataset stats for stricter control."
    )
