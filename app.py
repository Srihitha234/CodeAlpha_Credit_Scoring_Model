import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load("credit_scoring_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
feature_medians = joblib.load("feature_medians.pkl")
st.set_page_config(
    page_title="Credit Scoring System",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Scoring System")
st.write("Predict an individual's creditworthiness")

st.sidebar.header("Customer Details")

# Important inputs
ext_source_2 = st.sidebar.slider(
    "External Credit Score 2",
    0.0, 1.0, 0.50
)

ext_source_3 = st.sidebar.slider(
    "External Credit Score 3",
    0.0, 1.0, 0.50
)

amt_credit = st.sidebar.number_input(
    "Credit Amount",
    min_value=0.0,
    value=500000.0
)

amt_annuity = st.sidebar.number_input(
    "Annuity Amount",
    min_value=0.0,
    value=25000.0
)

days_birth = st.sidebar.number_input(
    "Days Birth",
    value=-12000
)

days_employed = st.sidebar.number_input(
    "Days Employed",
    value=-3000
)

days_registration = st.sidebar.number_input(
    "Days Registration",
    value=-4000
)

days_id_publish = st.sidebar.number_input(
    "Days ID Publish",
    value=-2500
)

days_last_phone_change = st.sidebar.number_input(
    "Days Last Phone Change",
    value=-1000
)

if st.button("Predict Creditworthiness"):

    # Create all features with default 0
    data = feature_medians.copy()

    # Fill important features
    if "EXT_SOURCE_2" in data:
        data["EXT_SOURCE_2"] = ext_source_2

    if "EXT_SOURCE_3" in data:
        data["EXT_SOURCE_3"] = ext_source_3

    if "AMT_CREDIT" in data:
        data["AMT_CREDIT"] = amt_credit

    if "AMT_ANNUITY" in data:
        data["AMT_ANNUITY"] = amt_annuity

    if "DAYS_BIRTH" in data:
        data["DAYS_BIRTH"] = days_birth

    if "DAYS_EMPLOYED" in data:
        data["DAYS_EMPLOYED"] = days_employed

    if "DAYS_REGISTRATION" in data:
        data["DAYS_REGISTRATION"] = days_registration

    if "DAYS_ID_PUBLISH" in data:
        data["DAYS_ID_PUBLISH"] = days_id_publish

    if "DAYS_LAST_PHONE_CHANGE" in data:
        data["DAYS_LAST_PHONE_CHANGE"] = days_last_phone_change

    df = pd.DataFrame([data])

    scaled_data = scaler.transform(df)

    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    st.subheader("Prediction Result")

    
    st.progress(float(probability))
    risk_percent = probability * 100
    if risk_percent >= 60:
        st.error("🔴 High Risk Customer")

    elif risk_percent >= 40:
        st.warning("🟡 Medium Risk Customer")

    else:
        st.success("🟢 Creditworthy Customer")

    st.metric(
        "Default Probability",
        f"{probability * 100:.2f}%"
    )