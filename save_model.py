# save_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import ADASYN
import xgboost as xgb
import pickle

# Load dataset
df = pd.read_csv("global_house_purchase_dataset.csv")

# One-hot encode categorical columns
ohe_col = ["country", "city", "property_type"]
df_encoded = pd.get_dummies(df, columns=ohe_col, prefix=ohe_col)

# Drop irrelevant columns
drop_col = [
    "property_id", "furnishing_status", "property_size_sqft", "price",
    "constructed_year", "previous_owners", "rooms", "bathrooms", "garage",
    "garden", "crime_cases_reported", "legal_cases_on_property", "customer_salary",
    "loan_amount", "loan_tenure_years", "monthly_expenses", "down_payment",
    "emi_to_income_ratio", "satisfaction_score", "neighbourhood_rating",
    "connectivity_score", "decision"
]
df_encoded = df_encoded.drop(columns=drop_col, errors="ignore")

# Combine original + encoded
df = pd.concat([df, df_encoded], axis=1)

# Ordinal encode furnishing_status
mapping = {"Unfurnished": 0, "Semi-Furnished": 1, "Fully-Furnished": 2}
df["furnishing_status"] = df["furnishing_status"].map(mapping)

# Drop original categorical columns
df = df.drop(columns=ohe_col, errors="ignore")

# Features and target
X = df.drop(columns=["decision", "property_id"], errors="ignore")
y = df["decision"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Log transform
log_col = ["price", "loan_amount", "down_payment"]
for col in log_col:
    if col in X_train.columns:
        X_train[col] = np.log(X_train[col] + 1)
        X_test[col] = np.log(X_test[col] + 1)

# Power transform (Yeo–Johnson)
if "emi_to_income_ratio" in X_train.columns:
    yeo = PowerTransformer(method="yeo-johnson")
    X_train["emi_to_income_ratio"] = yeo.fit_transform(X_train[["emi_to_income_ratio"]])
    X_test["emi_to_income_ratio"] = yeo.transform(X_test[["emi_to_income_ratio"]])
else:
    yeo = None

# Handle imbalance using ADASYN
ad = ADASYN()
X_train, y_train = ad.fit_resample(X_train, y_train)

# Train XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
model.fit(X_train, y_train)

# Save model + transformations
artifacts = {
    "model": model,
    "yeo": yeo,
    "columns": X.columns.tolist(),
    "mapping": mapping
}

with open("xgb_full_pipeline.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("✅ Model and transformations saved as xgb_full_pipeline.pkl")
