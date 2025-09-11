import streamlit as st
import joblib
import pandas as pd

# Load the saved model and label encoder
try:
    loaded_model = joblib.load('retrained_model.pkl')
    loaded_label_encoder = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'retrained_model.pkl' and 'label_encoder.pkl' are in the same directory.")
    st.stop()

# Define the expected features based on the training data
# You can get this from your X_train or X_test DataFrame columns
# For this example, I'll list them out. Make sure this matches your training data.
expected_features = [
    'Memory Recall (%)', 'Gait Speed (m/s)', 'Tremor Frequency (Hz)',
    'Speech Rate (wpm)', 'Reaction Time (ms)', 'Eye Movement Irregularities (saccades/s)',
    'Sleep Disturbance (scale 0-10)', 'Cognitive Test Score (MMSE)',
    'Blood Pressure (mmHg)', 'Cholesterol (mg/dL)', 'Diabetes', 'Severity',
    'Gender_Male'
]

st.title("Neurological Condition Prediction")

st.write("Enter the patient's characteristics to predict the neurological condition.")

# Create input fields for each feature
input_data = {}
for feature in expected_features:
    if feature in ['Memory Recall (%)', 'Speech Rate (wpm)', 'Reaction Time (ms)',
                   'Sleep Disturbance (scale 0-10)', 'Cognitive Test Score (MMSE)',
                   'Blood Pressure (mmHg)', 'Cholesterol (mg/dL)', 'Diabetes',
                   'Severity', 'Gender_Male']:
        input_data[feature] = st.number_input(f"Enter {feature}", value=0)
    elif feature in ['Gait Speed (m/s)', 'Tremor Frequency (Hz)', 'Eye Movement Irregularities (saccades/s)']:
        input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)


if st.button("Predict"):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Ensure the order of columns in input_df matches the training data
    input_df = input_df[expected_features]

    # Make prediction
    try:
        prediction_encoded = loaded_model.predict(input_df)
        prediction_label = loaded_label_encoder.inverse_transform(prediction_encoded)
        st.success(f"Predicted Neurological Condition: {prediction_label[0]}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
