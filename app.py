import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load model + preprocessing artifacts ---
with open("classifier2.pkl", "rb") as model_file:
    artifacts = pickle.load(model_file)

model = artifacts["model"]
yeo = artifacts["yeo"]
columns = artifacts["columns"]
mapping = artifacts["mapping"]

# --- Load dataset to extract categorical options ---
df = pd.read_csv("global_house_purchase_dataset.csv")
countries = sorted(df["country"].dropna().unique())
cities = sorted(df["city"].dropna().unique())
property_types = sorted(df["property_type"].dropna().unique())

# --- Streamlit UI setup ---
st.set_page_config(page_title="🏠 House Purchase Predictor", page_icon="🏠")
st.title("🏠 House Purchase Decision Predictor (XGBoost)")
st.write("Enter details to predict whether the buyer will **Buy or Not Buy** the house.")

# --- Furnishing ---
st.subheader("🏡 Furnishing Status")
furnishing_status_selected = st.selectbox("Select Furnishing Status", list(mapping.keys()), index=0)
input_data = {"furnishing_status": mapping[furnishing_status_selected]}

# --- Location & property type ---
st.subheader("🌍 Property Location & Type")
selected_country = st.selectbox("Country", countries, index=0)
selected_city = st.selectbox("City", cities, index=0)
selected_type = st.selectbox("Property Type", property_types, index=0)

# --- Numerical features ---
st.subheader("📊 Numerical Features")

# Default values that tend to make a person BUY
def smart_default(col):
    name = col.lower()
    if "income" in name: return 150000.0
    if "value" in name or "price" in name: return 200000.0
    if "loan" in name: return 100000.0
    if "credit" in name: return 780.0
    if "ratio" in name: return 0.2
    if "age" in name: return 35.0
    if "depend" in name: return 1.0
    if "room" in name: return 4.0
    if "bath" in name: return 3.0
    if "garage" in name or "garden" in name: return 1.0
    if "crime" in name: return 0.5
    if "constructed" in name or "year" in name: return 2019.0
    return 0.0

num_features = [
    col for col in columns
    if all(x not in col for x in ["country_", "city_", "property_type_", "furnishing_status"])
]

for col in num_features:
    default_val = float(smart_default(col))
    input_data[col] = st.number_input(f"{col}:", value=default_val, step=1.0)

# --- Prepare DataFrame ---
df_input = pd.DataFrame([input_data])

# --- Transformations ---
if "emi_to_income_ratio" in df_input.columns and yeo:
    df_input["emi_to_income_ratio"] = yeo.transform(df_input[["emi_to_income_ratio"]])

# --- One-Hot Encoding ---
for col in columns:
    if col.startswith("country_"):
        df_input[col] = 1 if col == f"country_{selected_country}" else 0
    elif col.startswith("city_"):
        df_input[col] = 1 if col == f"city_{selected_city}" else 0
    elif col.startswith("property_type_"):
        df_input[col] = 1 if col == f"property_type_{selected_type}" else 0
    elif col not in df_input.columns:
        df_input[col] = 0

# --- Align columns safely ---
df_input = df_input.reindex(columns=columns, fill_value=0)

# --- Prediction ---
if st.button("🔮 Predict Decision", type="primary"):
    pred = model.predict(df_input)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df_input)[0]
        confidence = float(max(proba))
        st.progress(confidence)
        st.info(f"Prediction Confidence: {confidence*100:.1f}%")
    else:
        confidence = None

    if hasattr(model, "classes_"):
        buy_label = max(model.classes_)
    else:
        buy_label = 1

    label = "✅ BUY" if pred == buy_label else "❌ NOT BUY"

    st.subheader(f"### 🧠 Model Prediction: {label}")
    st.caption(f"Confidence: {confidence*100:.1f}%" if confidence else "Confidence not available")

st.caption("Default values are tuned for a likely **BUY** case. Adjust to test NOT BUY scenarios.")
