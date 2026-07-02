import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
 
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")
 
# -----------------------------
# Feature metadata (standard heart.csv / UCI heart disease dataset)
# -----------------------------
FEATURE_INFO = {
    "age":      {"label": "Age",                              "type": "number",  "min": 1,   "max": 120, "default": 50},
    "sex":      {"label": "Sex",                               "type": "select",  "options": {"Male": 1, "Female": 0}},
    "cp":       {"label": "Chest Pain Type",                   "type": "select",  "options": {"Typical angina": 0, "Atypical angina": 1, "Non-anginal pain": 2, "Asymptomatic": 3}},
    "trestbps": {"label": "Resting Blood Pressure (mm Hg)",    "type": "number",  "min": 60,  "max": 250, "default": 120},
    "chol":     {"label": "Serum Cholesterol (mg/dl)",         "type": "number",  "min": 100, "max": 600, "default": 200},
    "fbs":      {"label": "Fasting Blood Sugar > 120 mg/dl",   "type": "select",  "options": {"No": 0, "Yes": 1}},
    "restecg":  {"label": "Resting ECG Results",               "type": "select",  "options": {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}},
    "thalach":  {"label": "Max Heart Rate Achieved",           "type": "number",  "min": 60,  "max": 250, "default": 150},
    "exang":    {"label": "Exercise Induced Angina",           "type": "select",  "options": {"No": 0, "Yes": 1}},
    "oldpeak":  {"label": "ST Depression (oldpeak)",           "type": "float",   "min": 0.0, "max": 10.0, "default": 1.0},
    "slope":    {"label": "Slope of Peak Exercise ST Segment", "type": "select",  "options": {"Upsloping": 0, "Flat": 1, "Downsloping": 2}},
    "ca":       {"label": "Number of Major Vessels (0-4)",     "type": "select",  "options": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}},
    "thal":     {"label": "Thalassemia",                       "type": "select",  "options": {"Normal": 1, "Fixed defect": 2, "Reversible defect": 3}},
}
 
 
@st.cache_data
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)
 
 
@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df.drop("target", axis=1)
    y = df["target"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
 
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
 
    return model, X, X_train, X_test, y_train, y_test, y_pred, accuracy
 
 
def main():
    st.title("❤️ Heart Disease Prediction")
    st.write(
        "This app trains a Logistic Regression model on the heart disease dataset "
        "and lets you predict whether a patient is likely to have heart disease."
    )
 
    # -----------------------------
    # Data loading
    # -----------------------------
    st.sidebar.header("1. Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload heart.csv", type=["csv"])
 
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        try:
            df = load_data("heart.csv")
            st.sidebar.info("Using bundled `heart.csv` from the app directory.")
        except FileNotFoundError:
            st.warning(
                "No dataset found. Please upload `heart.csv` (must contain the usual "
                "heart-disease columns plus a `target` column) using the sidebar."
            )
            st.stop()
 
    if "target" not in df.columns:
        st.error("The dataset must contain a `target` column (0 = no disease, 1 = disease).")
        st.stop()
 
    # -----------------------------
    # Train model
    # -----------------------------
    model, X, X_train, X_test, y_train, y_test, y_pred, accuracy = train_model(df)
 
    tab_predict, tab_explore, tab_performance = st.tabs(
        ["🔮 Predict", "📊 Explore Data", "📈 Model Performance"]
    )
 
    # -----------------------------
    # Tab 1: Prediction
    # -----------------------------
    with tab_predict:
        st.subheader("Enter Patient Details")
 
        cols = list(X.columns)
        input_values = {}
        col_layout = st.columns(2)
 
        for i, col in enumerate(cols):
            target_col = col_layout[i % 2]
            info = FEATURE_INFO.get(col)
 
            with target_col:
                if info is None:
                    # Fallback for any unexpected column: plain numeric input
                    input_values[col] = st.number_input(col, value=float(df[col].median()))
                elif info["type"] == "select":
                    choice = st.selectbox(info["label"], list(info["options"].keys()))
                    input_values[col] = info["options"][choice]
                elif info["type"] == "float":
                    input_values[col] = st.number_input(
                        info["label"], min_value=info["min"], max_value=info["max"],
                        value=info["default"], step=0.1
                    )
                else:
                    input_values[col] = st.number_input(
                        info["label"], min_value=info["min"], max_value=info["max"],
                        value=info["default"], step=1
                    )
 
        if st.button("Predict", type="primary"):
            input_df = pd.DataFrame([input_values])[cols]
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]
 
            if prediction == 1:
                st.error(f"⚠️ High risk of heart disease (probability: {proba:.1%})")
            else:
                st.success(f"✅ Low risk of heart disease (probability: {proba:.1%})")
 
    # -----------------------------
    # Tab 2: Data exploration
    # -----------------------------
    with tab_explore:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)
 
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Shape:**", df.shape)
            st.write("**Missing values:**")
            st.dataframe(df.isnull().sum().rename("missing"))
        with col2:
            st.write("**Summary statistics:**")
            st.dataframe(df.describe())
 
    # -----------------------------
    # Tab 3: Model performance
    # -----------------------------
    with tab_performance:
        st.subheader("Model Evaluation")
        st.metric("Accuracy", f"{accuracy:.2%}")
 
        col1, col2 = st.columns(2)
 
        with col1:
            st.write("**Confusion Matrix**")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
            st.pyplot(fig)
 
        with col2:
            st.write("**ROC Curve**")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
            st.pyplot(fig)
 
        st.write("**Feature Importance (Logistic Regression Coefficients)**")
        coef = pd.Series(model.coef_[0], index=X.columns).sort_values()
        fig, ax = plt.subplots(figsize=(8, 6))
        coef.plot(kind="barh", ax=ax)
        ax.set_xlabel("Coefficient Value")
        st.pyplot(fig)
 
        st.write("**Actual vs Predicted (first 10 test rows)**")
        result = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).head(10)
        st.dataframe(result, use_container_width=True)
 
 
if __name__ == "__main__":
    main()
