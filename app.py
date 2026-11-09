# app.py
import os
import runpy

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Predict", page_icon="🎓", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)


@st.cache_resource(show_spinner="Preparing model (first run only, please wait)...")
def prepare():
    """Load the model and data. If the files are missing (e.g. on Streamlit
    Cloud) or the saved model is incompatible, run train_model.py to rebuild."""
    try:
        return joblib.load("student_model.pkl"), pd.read_csv("student_data.csv")
    except Exception:
        runpy.run_path("train_model.py")
        return joblib.load("student_model.pkl"), pd.read_csv("student_data.csv")


try:
    model, data = prepare()
except Exception as exc:
    st.error("Could not prepare the model. Details below - please send a screenshot.")
    st.exception(exc)
    st.stop()

st.title("🎓 Student Performance Prediction")
st.caption("Classroom prototype for academic risk exploration  |  app version v2 (auto-train)")
st.warning(
    "Predictions are not diagnoses and must not be used as the sole basis "
    "for real student decisions."
)

c1, c2, c3 = st.columns(3)
c1.metric("Records", len(data))
c2.metric("Average final grade", round(data["G3"].mean(), 2))
c3.metric("Historical at-risk records", int((data["G3"] < 10).sum()))

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)

st.subheader("Predict uploaded student records")
st.write("Upload a CSV containing the same predictor columns as the training data.")

template = data.drop(columns=["G3"]).head(5)
st.download_button(
    "Download sample input CSV",
    template.to_csv(index=False),
    "sample_prediction_input.csv",
    "text/csv",
)

uploaded = st.file_uploader("Choose student CSV", type=["csv"])

if uploaded is not None:
    new_data = pd.read_csv(uploaded)
    required_columns = list(model.feature_names_in_)
    missing = [c for c in required_columns if c not in new_data.columns]

    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
    else:
        predictions = model.predict(new_data[required_columns])
        results = new_data.copy()
        results["Prediction"] = [
            "AT RISK" if value == 1 else "NOT AT RISK" for value in predictions
        ]

        st.subheader("Prediction results")
        st.dataframe(results, use_container_width=True)
        st.metric("Predicted at-risk records", int((predictions == 1).sum()))
        st.download_button(
            "Download prediction results",
            results.to_csv(index=False),
            "student_predictions.csv",
            "text/csv",
        )
