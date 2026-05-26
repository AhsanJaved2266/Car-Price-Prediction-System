import streamlit as st
import pandas as pd
import numpy as np
import joblib
 
# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)
 
# =========================================================
# LOAD MODEL & ENCODERS
# =========================================================
@st.cache_resource
def load_assets():
    model                = joblib.load('car_price_prediction_model.pkl')
    make_encoder         = joblib.load('make_encoder.pkl')
    model_encoder        = joblib.load('model_encoder.pkl')
    fuel_encoder         = joblib.load('fuel_encoder.pkl')
    transmission_encoder = joblib.load('transmission_encoder.pkl')
    drivetrain_encoder   = joblib.load('drivetrain_encoder.pkl')
    return model, make_encoder, model_encoder, fuel_encoder, transmission_encoder, drivetrain_encoder
 
# =========================================================
# LOAD & CLEAN DATA (for dropdown options)
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv('car details v4.csv')
 
    df = df[['Make', 'Model', 'Year', 'Kilometer',
             'Fuel Type', 'Transmission', 'Drivetrain', 'Engine', 'Price']]
    df = df.dropna()
 
    # Clean Make
    df['Make'] = df['Make'].str.strip().str.title()
 
    # Clean Model (base model only — same as notebook)
    df['Model'] = df['Model'].str.strip().str.lower()
    df['Model'] = df['Model'].str.replace('-', ' ')
    df['Model'] = df['Model'].str.replace('(', '').str.replace(')', '')
    df['Model'] = df['Model'].str.replace(',', '')
    df['Model'] = df['Model'].str.replace(r'\s+', ' ', regex=True)
    df['Model'] = df['Model'].str.split().str[0].str.title()
 
    # Clean Fuel Type
    df['Fuel Type'] = df['Fuel Type'].str.strip().str.title()
 
    # Clean Transmission
    df['Transmission'] = df['Transmission'].str.strip().str.title()
 
    # Clean Drivetrain
    df['Drivetrain'] = df['Drivetrain'].str.strip().str.upper()
 
    # Clean Engine (remove 'cc', convert to numeric)
    df['Engine'] = df['Engine'].astype(str)
    df['Engine'] = df['Engine'].str.replace('cc', '', regex=False).str.strip()
    df['Engine'] = pd.to_numeric(df['Engine'], errors='coerce')
    df = df.dropna()
 
    return df
 
# =========================================================
# INIT
# =========================================================
model, make_encoder, model_encoder, fuel_encoder, transmission_encoder, drivetrain_encoder = load_assets()
df = load_data()
 
# =========================================================
# UI
# =========================================================
st.title("🚗 Car Price Prediction System")
st.markdown("Enter the car details below to get an estimated price.")
st.markdown("---")
 
col1, col2 = st.columns(2)
 
with col1:
    make         = st.selectbox("Car Make",             sorted(df['Make'].unique()))
    year         = st.number_input("Year",              min_value=1990, max_value=2024, value=2020)
    fuel         = st.selectbox("Fuel Type",            sorted(df['Fuel Type'].unique()))
    drivetrain   = st.selectbox("Drivetrain",           sorted(df['Drivetrain'].unique()))
 
with col2:
    car_model    = st.selectbox("Car Model",            sorted(df['Model'].unique()))
    kilometer    = st.number_input("Kilometers Driven", min_value=0, value=30000, step=1000)
    transmission = st.selectbox("Transmission",         sorted(df['Transmission'].unique()))
    engine       = st.number_input("Engine (cc)",       min_value=500, max_value=10000, value=1800, step=100)
 
st.markdown("---")
 
# =========================================================
# PREDICTION
# =========================================================
if st.button("🔍 Predict Price", use_container_width=True):
    try:
        make_enc  = make_encoder.transform([make])[0]
        model_enc = model_encoder.transform([car_model])[0]
        fuel_enc  = fuel_encoder.transform([fuel])[0]
        trans_enc = transmission_encoder.transform([transmission])[0]
        drive_enc = drivetrain_encoder.transform([drivetrain])[0]
 
        input_data = np.array([[
            make_enc,
            model_enc,
            year,
            kilometer,
            fuel_enc,
            trans_enc,
            drive_enc,
            engine
        ]])
 
        predicted_price = model.predict(input_data)[0]
 
        st.success(f"💰 Estimated Car Price:  **{predicted_price:,.0f}**")
 
    except ValueError as e:
        st.error(f"Encoding Error: {e}. The selected value may not have been seen during training.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
 
st.markdown("---")
st.caption("Made by Talha Gillani | Ahsan Javed | Usaid Malik")
