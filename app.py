# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load model + preprocessing artifacts ---
with open("xgb_full_pipeline.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["model"]
yeo = artifacts.get("yeo", None)
columns = artifacts["columns"]
mapping = artifacts["mapping"]

# --- Load dataset for dropdowns ---
df = pd.read_csv("global_house_purchase_dataset.csv")
countries = sorted(df["country"].dropna().unique())
cities = sorted(df["city"].dropna().unique())
property_types = sorted(df["property_type"].dropna().unique())

# --- Streamlit UI setup ---
st.set_page_config(page_title="🏠 House Purchase Predictor", page_icon="🏠")
st.title("🏠 House Purchase Decision Predictor (XGBoost)")
st.write("Enter details to predict whether the buyer will **Buy or Not Buy** the house.")

# --- Categorical selections ---
st.subheader("🌍 Property Info")
selected_country = st.selectbox("Country", countries, index=0)
selected_city = st.selectbox("City", cities, index=0)
selected_type = st.selectbox("Property Type", property_types, index=0)
selected_furnish = st.selectbox("Furnishing Status", list(mapping.keys()), index=0)

# --- Numeric Inputs ---
st.subheader("📊 Buyer & Property Details")

def smart_default(col):
    name = col.lower()
    if "income" in name: return 20000.0
    if "value" in name or "price" in name: return 500000.0
    if "loan" in name: return 400000.0
    if "credit" in name: return 450.0
    if "ratio" in name: return 0.8
    if "age" in name: return 24.0
    if "depend" in name: return 4.0
    if "room" in name: return 1.0
    if "bath" in name: return 1.0
    if "garage" in name or "garden" in name: return 0.0
    if "crime" in name: return 20.0
    if "constructed" in name or "year" in name: return 1980.0
    return 0.0

input_data = {"furnishing_status": mapping[selected_furnish]}
num_features = [c for c in columns if all(x not in c for x in ["country_", "city_", "property_type_", "furnishing_status"])]

for col in num_features:
    input_data[col] = st.number_input(f"{col}:", value=float(smart_default(col)))

# --- Create input DataFrame ---
df_input = pd.DataFrame([input_data])

# --- Apply transformation ---
if "emi_to_income_ratio" in df_input.columns and yeo:
    df_input["emi_to_income_ratio"] = yeo.transform(df_input[["emi_to_income_ratio"]])

# --- One-hot encoding ---
for col in columns:
    if col.startswith("country_"):
        df_input[col] = 1 if col == f"country_{selected_country}" else 0
    elif col.startswith("city_"):
        df_input[col] = 1 if col == f"city_{selected_city}" else 0
    elif col.startswith("property_type_"):
        df_input[col] = 1 if col == f"property_type_{selected_type}" else 0
    elif col not in df_input.columns:
        df_input[col] = 0

df_input = df_input.reindex(columns=columns, fill_value=0)

# --- Prediction ---
if st.button("🔮 Predict Decision", type="primary"):
    pred = model.predict(df_input)[0]

    st.write(f"Raw model output: {pred}")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df_input)[0]
        st.write(f"Probabilities: {proba}")

    # Label based on class meaning
    label = "✅ BUY" if pred == 1 else "❌ NOT BUY"

    if hasattr(model, "predict_proba"):
        confidence = float(max(proba))
        st.progress(confidence)
        st.info(f"Prediction Confidence: {confidence*100:.1f}%")

    st.subheader(f"### 🧠 Model Prediction: {label}")

st.caption("Default values are tuned for a likely 'NOT BUY' case. Adjust features to test 'BUY' scenarios.")
