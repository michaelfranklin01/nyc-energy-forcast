import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_preprocessing import load_and_clean_data
from feature_engineering import create_features


def evaluate_model():
    # 1. Path to the trained model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'energy_rf.pkl')

    # Load the previously trained RandomForestRegressor
    rf_model = joblib.load(model_path)
    print("Loaded trained model from:", model_path)

    # 2. Process test data (2022) to get features (X_test) and target (y_test)
    print("Processing test data...")
    load_and_clean_data("merged_2021", "filtered_evt_EUI-2021.csv")
    print("Data Processed. \nGetting features...")
    X_test, y_test = create_features("merged_2021.geojson", "NYC_Weather_2021_monthly.csv")

    # 3. Generate predictions
    print("Features Generated. \nGenerating predictions...")
    y_pred = rf_model.predict(X_test)

    # 4. Compute your chosen metrics
    print("Predictions generated. \nEvaluating predictions...")
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_test = mean_absolute_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)

    print("Results: \nTest RMSE:", rmse_test)
    print("Test MAE: ", mae_test)
    print("Test R^2:", r2_test)

    # 5. (Optional) Save the predictions for further analysis
    results_df = X_test.copy()
    results_df['actual_energy'] = y_test
    results_df['predicted_energy'] = y_pred

    output_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'test_predictions.csv')
    results_df.to_csv(output_file, index=False)
    print(f"Predictions saved to: {output_file}")

    return rmse_test, mae_test, r2_test


if __name__ == "__main__":
    evaluate_model()
