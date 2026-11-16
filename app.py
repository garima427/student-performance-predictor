"""
Streamlit Dashboard for Student Performance Predictor.
"""

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

MODELS_DIR = Path("models")
FEATURES = ["study_hours", "attendance", "previous_score", "sleep_hours",
            "assignment_completion", "extracurricular"]


@st.cache_resource
def load_models():
    try:
        return {
            "Simple Linear Regression": joblib.load(MODELS_DIR / "simple_linear.joblib"),
            "Multiple Linear Regression": joblib.load(MODELS_DIR / "multiple_linear.joblib"),
            "KNN Regressor": joblib.load(MODELS_DIR / "knn.joblib"),
        }, joblib.load(MODELS_DIR / "scaler.joblib")
    except Exception as e:
        st.error(f"Error loading models: {e}. Run model_training.py first.")
        return None, None


@st.cache_data
def load_results():
    return pd.DataFrame([
        {"Model": "Simple Linear Regression", "MAE": 7.55, "MSE": 117.61, "RMSE": 10.84, "R²": 0.771},
        {"Model": "Multiple Linear Regression", "MAE": 8.08, "MSE": 126.30, "RMSE": 11.24, "R²": 0.754},
        {"Model": "KNN Regressor (K=7)", "MAE": 9.23, "MSE": 179.44, "RMSE": 13.40, "R²": 0.650},
    ])


def main():
    st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")
    st.title("🎓 Student Performance Predictor")
    st.markdown("Predict a student's final exam score based on academic and lifestyle factors.")

    models, scaler = load_models()
    if models is None:
        st.stop()

    page = st.sidebar.radio("Navigate", ["Predict", "Model Comparison", "EDA"])

    if page == "Predict":
        st.header("Enter Student Details")
        col1, col2 = st.columns(2)
        with col1:
            study_hours = st.slider("Study Hours per Day", 0.5, 10.0, 3.0, 0.1)
            attendance = st.slider("Attendance (%)", 0, 100, 80, 1)
            previous_score = st.slider("Previous Exam Score", 0, 100, 70, 1)
        with col2:
            sleep_hours = st.slider("Sleep Hours", 3.0, 10.0, 7.0, 0.5)
            assignment_completion = st.slider("Assignment Completion (%)", 0, 100, 75, 1)
            extracurricular = st.selectbox("Extracurricular Activities", ["No", "Yes"])

        model_name = st.selectbox("Choose Model", list(models.keys()))

        if st.button("Predict Final Score", type="primary"):
            input_data = pd.DataFrame([{
                "study_hours": study_hours,
                "attendance": attendance,
                "previous_score": previous_score,
                "sleep_hours": sleep_hours,
                "assignment_completion": assignment_completion,
                "extracurricular": 1 if extracurricular == "Yes" else 0,
            }])

            # Simple Linear uses only previous_score
            if "Simple" in model_name:
                feature_idx = FEATURES.index("previous_score")
                pred_input = input_data.iloc[:, [feature_idx]]
                # Simple linear was trained on scaled data for previous_score
                # But it was trained with index-based access - reconstruct properly
                pred_input = scaler.transform(input_data)[:, [feature_idx]]
            else:
                pred_input = scaler.transform(input_data)

            prediction = models[model_name].predict(pred_input)[0]
            prediction = max(0, min(100, prediction))  # Clamp to 0-100

            st.success(f"**Predicted Final Score: {prediction:.1f}/100**")
            if prediction >= 80:
                st.balloons()
                st.write("Excellent predicted performance! 🎉")
            elif prediction >= 60:
                st.write("Good performance expected. Keep it up! 💪")
            else:
                st.warning("Consider increasing study hours and attendance.")

    elif page == "Model Comparison":
        st.header("Model Comparison")
        results_df = load_results()
        st.dataframe(results_df, use_container_width=True)
        st.markdown("**Metrics explained:**")
        st.markdown("- **MAE** (Mean Absolute Error): Average absolute difference between predicted and actual scores")
        st.markdown("- **MSE** (Mean Squared Error): Average squared difference; penalizes larger errors more")
        st.markdown("- **RMSE** (Root Mean Squared Error): Square root of MSE; in same units as score")
        st.markdown("- **R²** (R-squared): Proportion of variance explained; higher is better (max 1.0)")
        st.info("Lower MAE/MSE/RMSE and higher R² indicate better model performance.")

    elif page == "EDA":
        st.header("Exploratory Data Analysis")
        for img in ["eda_final_score.png", "eda_study_hours.png", "eda_attendance.png",
                    "eda_previous_score.png", "eda_correlation.png", "eda_knn_k_values.png"]:
            if Path(img).exists():
                st.image(img, caption=img.replace("eda_", "").replace(".png", "").replace("_", " ").title())
            else:
                st.info(f"{img} not found. Run eda.py first.")


if __name__ == "__main__":
    main()