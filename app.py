import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Page Config (MUST BE FIRST)
# -----------------------------
st.set_page_config(
    page_title="Sleep Disorder Predictor",
    layout="centered"
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pipeline.pkl")
CSS_PATH = os.path.join(BASE_DIR, "style.css")

# -----------------------------
# Load CSS
# -----------------------------
def load_css():
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -----------------------------
# UI Header
# -----------------------------
st.title("Sleep Risk Analyzer")

st.markdown(
"""
### Data-Driven Sleep Health Assessment

This application evaluates lifestyle and physiological indicators to estimate the risk of common sleep disorders, including **Insomnia** and **Sleep Apnea**.

The system demonstrates a complete machine learning workflow — from data preprocessing and feature engineering to model training, evaluation, and interactive deployment.
"""
)

# -----------------------------
# INPUT SECTION
# -----------------------------
st.subheader("Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 18, 100, 30)
    occupation = st.text_input("Occupation", "Engineer")
    sleep_duration = st.number_input("Sleep Duration (hrs)", 0.0, 12.0, 7.0)
    quality = st.slider("Quality of Sleep (1-10)", 1, 10, 6)

with col2:
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    bmi = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
    heart_rate = st.number_input("Heart Rate", 40, 120, 70)
    systolic = st.number_input("Systolic BP", 90, 200, 120)
    diastolic = st.number_input("Diastolic BP", 60, 120, 80)
    steps_per_day = st.number_input("Daily Steps", 0, 20000, 4000)
    activity_index = steps_per_day * 30

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict"):

    input_data = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Occupation": occupation,
        "Sleep Duration": sleep_duration,
        "Quality of Sleep": quality,
        "Stress Level": stress,
        "BMI Category": bmi,
        "Heart Rate": heart_rate,
        "Systolic": systolic,
        "Diastolic": diastolic,
        "Activity_Index": activity_index
    }])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    confidence = max(probabilities)

    # -----------------------------
    # RESULT SECTION
    # -----------------------------
    st.header("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Condition", prediction)

    with col2:
        st.metric("Confidence", f"{confidence*100:.2f}%")

    st.progress(int(confidence * 100))

    # Risk Badge
    if confidence > 0.85:
        color = "#ff4b4b"
        level = "High Risk"
    elif confidence > 0.65:
        color = "#ffa500"
        level = "Moderate Risk"
    else:
        color = "#00cc96"
        level = "Low Risk"

    st.markdown(
        f"""
        <div style='
            padding: 15px;
            border-radius: 10px;
            background-color: {color};
            color: white;
            font-size: 18px;
            text-align: center;
            font-weight: bold;
        '>
            {level}
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------
    # Probability Chart
    # -----------------------------
    st.subheader(" Prediction Probabilities")

    prob_df = pd.DataFrame({
        "Condition": model.classes_,
        "Probability": probabilities
    })

    st.bar_chart(prob_df.set_index("Condition"))

    # -----------------------------
    # HEALTH INSIGHTS
    # -----------------------------
    st.subheader(" Health Insights")

    if stress >= 8:
        st.warning("High stress detected. Consider stress management techniques.")

    if sleep_duration < 6:
        st.warning("Low sleep duration. Recommended 7–8 hours.")

    if bmi == "Obese":
        st.warning("BMI indicates obesity. Lifestyle changes recommended.")

    if heart_rate > 100:
        st.warning("High resting heart rate detected.")

    if systolic > 140 or diastolic > 90:
        st.warning("Elevated blood pressure observed.")

    # -----------------------------
    # FEATURE IMPORTANCE
    # -----------------------------
    st.subheader("Top Influencing Factors")

    try:
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        importances = model.named_steps["model"].feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })

        # Clean feature names
        importance_df["Feature"] = (
            importance_df["Feature"]
            .str.replace("num__", "", regex=False)
            .str.replace("cat__", "", regex=False)
            .str.replace("_", " ", regex=False)
            .str.title()
        )

        importance_df = (
            importance_df
            .sort_values(by="Importance", ascending=False)
            .head(8)
            .iloc[::-1]
        )

        st.bar_chart(
            importance_df.set_index("Feature")
        )

    except Exception:
        st.info("Feature importance not available.")

    st.markdown(
    "<small>For educational and predictive modeling purposes only.</small>",
    unsafe_allow_html=True
    )