import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# LOAD DATASET
# -----------------------------

df = pd.read_csv("car details v4.csv")

# -----------------------------
# CLEAN DATA
# -----------------------------

# Remove rows where Price is missing
df = df.dropna(subset=['Price'])

# Convert Engine column safely
df['Engine'] = df['Engine'].astype(str)

df['Engine'] = df['Engine'].str.replace(' CC', '', regex=False)

df['Engine'] = pd.to_numeric(

    df['Engine'],

    errors='coerce'

)

# Fill missing engine values
df['Engine'] = df['Engine'].fillna(

    df['Engine'].median()

)

# Fill missing categorical values

categorical_columns = [

    'Make',
    'Model',
    'Fuel Type',
    'Transmission',
    'Drivetrain'

]

for col in categorical_columns:

    df[col] = df[col].fillna(

        df[col].mode()[0]

    )

# Fill missing Kilometer
df['Kilometer'] = df['Kilometer'].fillna(

    df['Kilometer'].median()

)

# -----------------------------
# FEATURES & TARGET
# -----------------------------

features = [

    'Make',
    'Model',
    'Year',
    'Kilometer',
    'Fuel Type',
    'Transmission',
    'Engine',
    'Drivetrain'

]

X = df[features]

y = df['Price']

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)

# -----------------------------
# PREPROCESSOR
# -----------------------------

categorical_features = [

    'Make',
    'Model',
    'Fuel Type',
    'Transmission',
    'Drivetrain'

]

numerical_features = [

    'Year',
    'Kilometer',
    'Engine'

]

preprocessor = ColumnTransformer(

    transformers=[

        (

            'num',
            StandardScaler(),
            numerical_features

        ),

        (

            'cat',
            OneHotEncoder(handle_unknown='ignore'),
            categorical_features

        )

    ]

)

# -----------------------------
# MODEL
# -----------------------------

model = Pipeline([

    ('preprocessor', preprocessor),

    (

        'regressor',

        RandomForestRegressor(

            n_estimators=100,
            random_state=42

        )

    )

])

model.fit(X_train, y_train)

# -----------------------------
# STREAMLIT UI
# -----------------------------

st.title("Car Price Prediction System")

st.write(

    "Machine Learning Based Used Car Price Predictor"

)

# -----------------------------
# USER INPUTS
# -----------------------------

make = st.selectbox(

    "Select Make",

    sorted(df['Make'].unique())

)

model_name = st.selectbox(

    "Select Model",

    sorted(df['Model'].unique())

)

year = st.number_input(

    "Enter Year",

    min_value=1990,
    max_value=2025,
    value=2020

)

kilometer = st.number_input(

    "Kilometers Driven",

    min_value=0.0,
    value=50000.0

)

fuel_type = st.selectbox(

    "Fuel Type",

    sorted(df['Fuel Type'].unique())

)

transmission = st.selectbox(

    "Transmission",

    sorted(df['Transmission'].unique())

)

engine = st.number_input(

    "Engine CC",

    min_value=500.0,
    value=1200.0

)

drivetrain = st.selectbox(

    "Drivetrain",

    sorted(df['Drivetrain'].unique())

)

# -----------------------------
# PREDICTION
# -----------------------------

if st.button("Predict Price"):

    input_data = pd.DataFrame({

        'Make': [make],
        'Model': [model_name],
        'Year': [year],
        'Kilometer': [kilometer],
        'Fuel Type': [fuel_type],
        'Transmission': [transmission],
        'Engine': [engine],
        'Drivetrain': [drivetrain]

    })

    prediction = model.predict(input_data)

    st.success(

        f"Predicted Car Price: {prediction[0]:,.2f}"

    )
