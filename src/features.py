from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def get_feature_pipeline():
    """
    Returns a preprocessing pipeline for transforming the input features:
    - Numerical features are scaled using MinMaxScaler.
    - Categorical features are one-hot encoded.
    """
    numeric_features = ['years_old', 'km_driven']
    categorical_features = ['company', 'model', 'fuel', 'transmission', 'owner']

    preprocessor = ColumnTransformer(transformers=[
        ('num', MinMaxScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

    return preprocessor

def get_feature_columns():
    """
    Returns the list of features used for training and prediction.
    """
    return ['company', 'model', 'years_old', 'km_driven', 'fuel', 'transmission', 'owner']
