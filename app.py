# app.py
import streamlit as st
import pandas as pd
import logging
from steps.feature_engineering_step import feature_engineering_step
from steps.outlier_detection_step import outlier_detection_step
from steps.model_loader import load_latest_pipeline

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

st.title("Online Shopping Prediction")

# User input form
with st.form("input_form"):
    Administrative = st.number_input("Administrative", value=1)
    Administrative_Duration = st.number_input("Administrative_Duration", value=0)
    Informational = st.number_input("Informational", value=0)
    Informational_Duration = st.number_input("Informational_Duration", value=0)
    ProductRelated = st.number_input("ProductRelated", value=20)
    ProductRelated_Duration = st.number_input("ProductRelated_Duration", value=200)
    BounceRates = st.number_input("BounceRates", value=0.0)
    ExitRates = st.number_input("ExitRates", value=0.0)
    PageValues = st.number_input("PageValues", value=0.0)
    SpecialDay = st.number_input("SpecialDay", value=0.0)
    Month = st.text_input("Month", value="Feb")
    VisitorType = st.selectbox("VisitorType", ["Returning_Visitor", "New_Visitor"])
    Weekend = st.selectbox("Weekend", [0, 1])
    
    submitted = st.form_submit_button("Predict")

if submitted:
    # Create DataFrame from input
    raw_row = pd.DataFrame([{
        "Administrative": Administrative,
        "Administrative_Duration": Administrative_Duration,
        "Informational": Informational,
        "Informational_Duration": Informational_Duration,
        "ProductRelated": ProductRelated,
        "ProductRelated_Duration": ProductRelated_Duration,
        "BounceRates": BounceRates,
        "ExitRates": ExitRates,
        "PageValues": PageValues,
        "SpecialDay": SpecialDay,
        "Month": Month,
        "VisitorType": VisitorType,
        "Weekend": Weekend
    }])

    logging.info("Raw input received:")
    logging.info(raw_row)

    # Preprocessing
    engineered_01 = feature_engineering_step(raw_row, strategy="binary_encoding", features=["Revenue", "Weekend"])
    engineered_02 = feature_engineering_step(engineered_01, strategy="month_season_encoding", features=["Month"])
    engineered_03 = feature_engineering_step(engineered_02, strategy="onehot_encoding", features=["Month", "VisitorType"])
    engineered_04 = feature_engineering_step(
        engineered_03,
        strategy="log",
        features=[
            "ProductRelated_Duration",
            "ProductRelated",
            "Informational_Duration",
            "Informational",
            "Administrative_Duration",
            "Administrative",
            "PageValues",
        ],
    )
    engineered_05 = outlier_detection_step(
        engineered_04,
        strategy="zscore",
        features=[
            "Informational",
            "Informational_Duration",
            "Administrative",
            "Administrative_Duration",
            "ProductRelated",
            "ProductRelated_Duration",
            "PageValues",
            "BounceRates",
        ],
    )

    # Load pipeline
    pipeline = load_latest_pipeline()

    # Align features
    if hasattr(pipeline, "feature_names_in_"):
        engineered_05 = engineered_05.reindex(columns=pipeline.feature_names_in_, fill_value=0)

    # Make prediction
    prediction = pipeline.predict(engineered_05)
    probabilities = pipeline.predict_proba(engineered_05) if hasattr(pipeline, "predict_proba") else None

    st.success(f"Prediction: {prediction[0]}")
    if probabilities is not None:
        st.info(f"Probabilities: {probabilities[0]}")
