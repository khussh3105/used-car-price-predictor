# 🚗 Used Car Price Predictor

An AI-powered web application that predicts the resale price of used cars based on key attributes such as brand, age, fuel type, and transmission. Built using **XGBoost** and **Streamlit**, supported by a robust, modular machine learning pipeline.

---

## 🔍 Project Overview

This project offers a smart and practical solution for estimating the fair resale value of used cars, making use of historical data and modern machine learning techniques.

* Real-world data cleaned and processed from **CarDekho** datasets.
* Feature-engineered pipeline using **scikit-learn** transformers.
* XGBoost regressor model with tuned hyperparameters for performance.
* **Streamlit**-based interactive web UI for end-user predictions.

---

## 💻 Tech Stack

* **Python 3.10**
* **pandas**, **scikit-learn**, **XGBoost**
* **Streamlit** for web interface
* **Joblib** for model serialization
* **Git + GitHub** for version control

---

## 🧱 Project Structure

```bash
used_car_price_predictor/
├── app.py                    # Streamlit app interface
├── main.py                   # Model training + evaluation
├── models/
│   ├── xgb_model.pkl         # Trained XGBoost model
│   └── encoder.pkl           # ColumnTransformer pipeline
├── data/
│   ├── car_details_from_car_dekho.csv
│   └── Car_details_v3.csv
├── src/
│   ├── cleaning.py           # Data cleaning functions
│   ├── features.py           # Feature engineering logic
│   └── model.py              # Model training and evaluation
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore                # Ignored files and folders
```

---

## 🔧 Installation & Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/khussh3105/used-car-price-predictor.git
cd used_car_price_predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python main.py
```

### 4. Launch the app

```bash
streamlit run app.py
```

---

## 📊 Model Performance

| Metric | Score    |
| ------ | -------- |
| RMSE   | **0.03** |
| MAE    | **0.02** |
| R²     | **0.87** |

Model performance was measured on a held-out test dataset using the cleaned and preprocessed input features.

---

## 🧐 Challenges Faced

* Cleaning and standardizing scraped real-world car data with inconsistencies
* Building a scalable, modular machine learning pipeline
* Avoiding overfitting while tuning XGBoost hyperparameters
* Designing a user interface that adapts dynamically to car brands and models

---

## ✨ Future Enhancements

* Add support for additional features such as car color, insurance status, and more detailed ownership history
* Enable real-time logging of user predictions for insights
* Deploy the application on a cloud platform (e.g., Heroku, Render) with versioned API endpoints
* Improve model performance with advanced ensemble methods
* Integrate SHAP or LIME for model explainability
* Add a contact form or feedback mechanism in the Streamlit UI

---

## 📅 Live App

Try the live version here: [Used Car Price Predictor](https://used-car-price-predictor-xtakff84wakedxtmxa82ax.streamlit.app/)

---

## 🧑‍💻 Author

Built with precision by **Khush Samir Kothari**
Pursuing Masters of Computer Science at University of Sydney (USYD)
