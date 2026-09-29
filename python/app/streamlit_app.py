from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

SRC = Path(__file__).resolve().parent.parent / "src"
model = joblib.load(SRC / "model.joblib")
columns = joblib.load(SRC / "columns.joblib")

st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("Customer Churn Prediction")
st.caption("XGBoost model trained on the Telco Customer Churn dataset")

with st.sidebar:
    st.header("Customer profile")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.slider("Monthly charges (€)", 18, 120, 70)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    tech = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
    senior = st.checkbox("Senior citizen")

row = pd.DataFrame(0.0, index=[0], columns=columns)

def set_col(name, value=1.0):
    if name in row.columns:
        row.at[0, name] = value

set_col("tenure", tenure)
set_col("MonthlyCharges", monthly)
set_col("TotalCharges", tenure * monthly)
set_col("SeniorCitizen", 1 if senior else 0)
set_col(f"Contract_{contract}")
set_col(f"InternetService_{internet}")
set_col(f"PaymentMethod_{payment}")
set_col(f"TechSupport_{tech}")

prob = float(model.predict_proba(row)[0, 1])

col1, col2 = st.columns(2)
with col1:
    st.metric("Churn probability", f"{prob:.0%}")
    st.progress(prob)
    if prob >= 0.6:
        st.error("High risk")
    elif prob >= 0.3:
        st.warning("Medium risk")
    else:
        st.success("Low risk")

with col2:
    st.subheader("Recommended action")
    if prob >= 0.3:
        if contract == "Month-to-month":
            st.write("• Offer a discount for a 1-year contract.")
        if tech == "No":
            st.write("• Offer free tech support for 3 months.")
        if payment == "Electronic check":
            st.write("• Encourage switching to automatic payment.")
        st.write("• Personal call from the retention team.")
    else:
        st.write("• No action needed. Keep the customer satisfied.")

st.subheader("What drives churn in the model")
imp = pd.Series(model.feature_importances_, index=columns).nlargest(10)
st.bar_chart(imp)