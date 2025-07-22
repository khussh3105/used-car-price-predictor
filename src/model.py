from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import numpy as np

def train_model(X, y):
    """
    Trains an XGBoost Regressor using GridSearchCV and returns the best model along with test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(objective='reg:squarederror', random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    return best_model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model and prints RMSE, MAE, and R² Score on normalized values.
    """
    y_pred = model.predict(X_test)

    y_test_norm = y_test / y_test.max()
    y_pred_norm = y_pred / y_test.max()

    mae = mean_absolute_error(y_test_norm, y_pred_norm)
    mse = mean_squared_error(y_test_norm, y_pred_norm)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_norm, y_pred_norm)

    print(f"RMSE = {rmse:.2f}")
    print(f"MAE = {mae:.2f}")
    print(f"R² Score = {r2:.2f}")
