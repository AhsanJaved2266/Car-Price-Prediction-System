import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and encoders
model = joblib.load('car_price_prediction_model.pkl')
make_encoder = joblib.load('make_encoder.pkl')
model_encoder = joblib.load('model_encoder.pkl')
fuel_encoder = joblib.load('fuel_encoder.pkl')
transmission_encoder = joblib.load('transmission_encoder.pkl')
drivetrain_encoder = joblib.load('drivetrain_encoder.pkl')

# Load dataset to get dropdown options
df = pd.read_csv('car details v4.csv')

# --- Data Cleaning (same as notebook) ---
df = df[['Make','Model','Year','Kilometer','Fuel Type','Transmission','Drivetrain','Engine','Price']]
df = df.dropna()
df['Make'] = df['Make'].str.strip().str.title()
df['Fuel Type'] = df['Fuel Type'].str.strip().str.title()
df['Transmission'] = df['Transmission'].str.strip().str.title()
df['Drivetrain'] = df['Drivetrain'].str.strip().str.upper()
df['Model'] = df['Model'].str.strip().str.lower()
df['Model'] = df['Model'].str.replace(r'[-(),]', '', regex=True)
df['Model'] = df['Model'].str.split().str[0].str.title()

# --- App UI ---
st.title("🚗 Car Price Prediction System")
st.markdown("Fill in the details below to get an estimated car price.")

col1, col2 = st.columns(2)

with col1:
    make = st.selectbox("Car Make", sorted(df['Make'].unique()))
    year = st.number_input("Year", min_value=1990, max_value=2024, value=2020)
    fuel = st.selectbox("Fuel Type", sorted(df['Fuel Type'].unique()))
    drivetrain = st.selectbox("Drivetrain", sorted(df['Drivetrain'].unique()))

with col2:
    car_model = st.selectbox("Car Model", sorted(df['Model'].unique()))
    kilometer = st.number_input("Kilometers Driven", min_value=0, value=30000)
    transmission = st.selectbox("Transmission", sorted(df['Transmission'].unique()))
    engine = st.number_input("Engine Capacity (L)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)

if st.button("Predict Price"):
    try:
        make_enc = make_encoder.transform([make])[0]
        model_enc = make_encoder.transform([car_model])[0]  # use model_encoder if separate
        fuel_enc = fuel_encoder.transform([fuel])[0]
        trans_enc = transmission_encoder.transform([transmission])[0]
        drive_enc = drivetrain_encoder.transform([drivetrain])[0]

        input_data = np.array([[make_enc, model_enc, year, kilometer,
                                 fuel_enc, trans_enc, drive_enc, engine]])

        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimated Car Price: **{prediction:,.0f}**")
    except Exception as e:
        st.error(f"Error: {e}. Make sure input values were seen during training.")