import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Import feature engineering function
from feature_engineering import create_features


def train_model():
    # Load training features and target (2021 data)
    X, y = create_features()

    # Split into training and validation sets (e.g., 80/20 split)
    # TODO: analyze the model output to find which factors most strongly predict energy consumption
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest model with basic parameters
    # (need to improve this over time and analyze our data more)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    rf_model.fit(X_train, y_train)

    # Evaluate on the validation set
    y_val_pred = rf_model.predict(X_val)
    rmse_val = mean_squared_error(y_val, y_val_pred, squared=False)
    print("Validation RMSE:", rmse_val)

    # Save the trained model for later use on testing data (2022)
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'energy_model_rf.pkl')
    joblib.dump(rf_model, model_path)
    print("Model saved to:", model_path)

    return rf_model


if __name__ == "__main__":
    model = train_model()
