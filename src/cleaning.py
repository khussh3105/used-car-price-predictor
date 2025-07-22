import pandas as pd
from datetime import datetime

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().dropna()
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    if 'year' in df.columns:
        current_year = datetime.now().year
        df['years_old'] = current_year - df['year']

    if 'name' in df.columns:
        df['company'] = df['name'].str.split().str[0]
        df['model'] = df['name'].str.split().str[1:].apply(lambda x: ' '.join(x))

    if 'km_driven' in df.columns:
        df['km_driven'] = df['km_driven'].astype(str).str.replace(',', '').astype(int)

    for col in ['fuel', 'transmission', 'owner', 'company', 'model']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    drop_cols = ['name', 'year', 'torque', 'engine', 'mileage', 'max_power', 'seats']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    if 'price' in df.columns:
        df = df[df['price'] > 0]
        df = df.rename(columns={'price': 'selling_price'})
    elif 'selling_price' in df.columns:
        df = df[df['selling_price'] > 0]
    else:
        raise ValueError("Neither 'price' nor 'selling_price' column found in dataset.")

    expected_cols = ['company', 'model', 'years_old', 'km_driven', 'fuel', 'transmission', 'owner', 'selling_price']
    df = df[[col for col in expected_cols if col in df.columns]]

    return df
