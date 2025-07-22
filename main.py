import pandas as pd
from src.cleaning import clean_data
from src.features import get_feature_columns, get_feature_pipeline
from src.model import train_model, evaluate_model
import joblib 

def load_data():
    df1 = pd.read_csv("data/car_details_from_car_dekho.csv")
    df2 = pd.read_csv("data/Car_details_v3.csv")
    combined_df = pd.concat([df1, df2], ignore_index=True)
    return clean_data(combined_df)

def main():
    df = load_data()

    if 'selling_price' not in df.columns:
        raise ValueError("Target column 'selling_price' not found after cleaning.")

    feature_columns = get_feature_columns()
    X = df[feature_columns]
    y = df['selling_price']

    pipeline = get_feature_pipeline()
    X_transformed = pipeline.fit_transform(X)

    # Dump the fitted encoder
    joblib.dump(pipeline, "models/encoder.pkl")

    model, X_test, y_test = train_model(X_transformed, y)

    # Dump the model
    joblib.dump(model, "models/xgb_model.pkl")

    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()

