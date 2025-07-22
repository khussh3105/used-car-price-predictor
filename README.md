The goal of this project is to develop a machine learning model that can reliably forecast used car prices based on a variety of factors, including mileage, engine size, fuel type, manufacturing year, and transmission type. 

Two CarDekho datasets, obtained through Kaggle, are used in the project:
- car_details_from_car_dekho.csv
- Car_details_v3.csv

A more thorough understanding of the data is made possible by the overlapping but distinct features found in these datasets.

Features
Preprocessing and data cleaning; feature engineering for engine, mileage, and category encodings
XGBoostRegressor and GridSearchCV with Hyper-Parameters is used for model training. RMSE, MAE, and R2 Score are used for evaluation.
Streamlit was used to create the interactive prediction interface. 


Dataset
Kaggle source is https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho


License
The only goals of this project are demonstration and education. The original author owns the dataset on Kaggle.