🚗 Car Price Prediction System
A Machine Learning web application that predicts the price of a car based on user-provided features. Built with Python, Scikit-learn, and deployed using Streamlit.

👨‍💻 Made By
NameStudent IDTalha Gillani2510158Ahsan Javed2510150Usaid Malik2510159

🌐 Live Demo
👉 Click here to open the app https://car-price-prediction-system-kfef4eykzswqwwxyggftrk.streamlit.app/


📌 Project Overview
This project predicts the selling price of a used car based on the following features:

🏷️ Car Make
🚘 Car Model
📅 Year of Manufacture
📍 Kilometers Driven
⛽ Fuel Type
⚙️ Transmission
🔧 Drivetrain
🔩 Engine Capacity (cc)


🧠 Machine Learning

Algorithm: Random Forest Regressor
Library: Scikit-learn
Encoding: Label Encoding for all categorical features
Evaluation Metrics: MAE, MSE, RMSE, R² Score


📂 Project Structure
car-price-prediction-system/
│
├── app.py                             # Streamlit web app
├── CAR_PRICE_PREDICTION.ipynb         # Jupyter Notebook (full ML pipeline)
├── car details v4.csv                 # Dataset
├── requirements.txt                   # Python dependencies
│
├── car_price_prediction_model.pkl     # Trained Random Forest model
├── make_encoder.pkl                   # Label encoder for Car Make
├── model_encoder.pkl                  # Label encoder for Car Model
├── fuel_encoder.pkl                   # Label encoder for Fuel Type
├── transmission_encoder.pkl           # Label encoder for Transmission
└── drivetrain_encoder.pkl             # Label encoder for Drivetrain

⚙️ How to Run Locally
1. Clone the repository
bashgit clone https://github.com/your-username/car-price-prediction-system.git
cd car-price-prediction-system
2. Install dependencies
bashpip install -r requirements.txt
3. Run the app
bashstreamlit run app.py

📦 Requirements
streamlit
pandas
numpy
scikit-learn
joblib

📊 Dataset

File: car details v4.csv
Contains details of used cars including make, model, year, kilometers driven, fuel type, transmission, drivetrain, engine capacity, and price.

📜 License
This project was developed as an academic project. Feel free to use it for learning purposes.
