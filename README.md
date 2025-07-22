🚗 Used Car Price Predictor
An AI-powered web application that predicts the resale price of used cars based on key attributes such as brand, age, fuel type, and transmission. Built using XGBoost and Streamlit, with a robust ML pipeline.

🔍 Project Overview
This project provides a solution for estimating the fair market price of a used car based on key features. It includes:

Cleaned real-world data scraped from CarDekho

A feature-engineered pipeline using scikit-learn

XGBoost regression model with hyperparameter tuning

A Streamlit web interface for interactive predictions

✅ Key Features
Predict used car resale prices based on:

Car company, model, age, kilometers driven, fuel type, transmission, and ownership history

Preprocessed and feature-engineered using consistent pipelines

Trained and evaluated using robust metrics (RMSE, MAE, R²)

Web UI for interactive input and predictions


🧱 Project Structure
used_car_price_predictor/
├── app.py                    # Streamlit app
├── main.py                   # Training + evaluation script
├── models/
│   ├── xgb_model.pkl         # Trained XGBoost model
│   └── encoder.pkl           # Preprocessing pipeline
├── data/
│   ├── car_details_from_car_dekho.csv
│   └── Car_details_v3.csv
├── src/
│   ├── cleaning.py           # Data cleaning
│   ├── features.py           # Feature transformations
│   └── model.py              # Model pipeline
├── requirements.txt
├── README.md
└── .gitignore

🔧 Installation & Running Locally

1. Clone the repository:
git clone https://github.com/khussh3105/used-car-price-predictor.git
cd used_car_price_predictor

2. Install dependencies:
pip install -r requirements.txt

3. Train the model:
python main.py

4. Launch the app:
streamlit run app.py


📊 Model Performance
| Metric | Score    |
| ------ | -------- |
| RMSE   | **0.03** |
| MAE    | **0.02** |
| R²     | **0.87** |


🧠 Challenges Faced
Dealing with inconsistencies in scraped real-world car data

Building a clean, reusable ML pipeline for production-ready predictions

Avoiding overfitting during hyperparameter tuning

Designing a UI that supports multiple car brands and models smoothly

👨‍💻 Author
Built with precision by Khush Samir Kothari 
Masters of Computer Science | University of Sydney.
