import streamlit as st
import pandas as pd
import joblib
import locale
from src.cleaning import clean_data
from src.features import get_feature_columns

# Set locale for Indian currency formatting
locale.setlocale(locale.LC_ALL, 'en_IN.UTF-8')

# --- Load and clean data ---
df_raw = pd.read_csv("data/car_details_from_car_dekho.csv")
df = clean_data(df_raw.copy())

# --- Dropdown values ---
companies = sorted(df['company'].unique())
models_by_company = {
    company: sorted(df[df['company'] == company]['model'].unique())
    for company in companies
}
fuel_types = sorted(df['fuel'].unique())
transmissions = sorted(df['transmission'].unique())
ownerships = sorted(df['owner'].unique())

# --- Load trained model and encoder pipeline ---
model = joblib.load("models/xgb_model.pkl")
encoder = joblib.load("models/encoder.pkl")  # This should be the fitted ColumnTransformer

# --- Streamlit UI ---
st.title("Used Car Price Predictor")
st.markdown("Enter the details of the used car below to estimate its selling price:")

selected_company = st.selectbox("Select Car Company", companies)
available_models = models_by_company[selected_company]
selected_model = st.selectbox("Select Car Model", available_models)

years_old = st.number_input("Years Old", min_value=0, max_value=50, value=5, step=1)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=30000, step=1000)
fuel = st.selectbox("Fuel Type", fuel_types)
transmission = st.selectbox("Transmission Type", transmissions)
owner = st.selectbox("Ownership Status", ownerships)

if st.button("Predict Price"):
    try:
        input_dict = {
            'company': selected_company.lower(),
            'model': selected_model.lower(),
            'years_old': years_old,
            'km_driven': km_driven,
            'fuel': fuel.lower(),
            'transmission': transmission.lower(),
            'owner': owner.lower()
        }

        input_df = pd.DataFrame([input_dict])[get_feature_columns()]
        input_encoded = encoder.transform(input_df)
        prediction = model.predict(input_encoded)[0]
        formatted_price = f"₹ {locale.format_string('%d', int(prediction), grouping=True)}"

        st.success(f"Estimated Selling Price: **{formatted_price}**")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")