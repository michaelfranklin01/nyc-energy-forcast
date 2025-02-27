# NYC Energy Forecast Project

This repository contains a machine learning project to forecast energy consumption for NYC buildings. The project integrates building attributes, geospatial footprints, and weather data to generate features for training regression models. The ultimate goal is to predict building energy usage accurately based on these factors.

## Project Overview

The project performs the following key steps:
- **Data Preprocessing:**  
  - Reads raw data from multiple CSV files including building footprints, energy usage, and weather data.
  - Merges building and energy datasets using a common identifier (BIN).
  - Cleans the data and creates a processed GeoJSON file (`merged_2021.geojson`) for 2021.

- **Feature Engineering:**  
  - Computes derived features such as building age and energy intensity (energy per square foot).
  - Applies one-hot encoding to categorical features (e.g., building type).
  - Integrates external weather data (e.g., average monthly temperature) as additional predictors.

- **Model Training and Evaluation:**  
  - Uses 2021 data as training data and 2022 as testing data.
  - Trains a Random Forest regressor (with scikit-learn) as a baseline model.
  - Evaluates the model using metrics such as RMSE, MAE, and R².
  - Saves the trained model for later predictions and further improvements.

## Project Structure

## Installation

1. **Install Anaconda/Miniconda:**  
   [Download and install Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

2. **Clone the Repository:**
   ```
   git clone https://github.com/michaelfranklin01/nyc-energy-forcast.git
   cd nyc_energy_forecast
   
3. Create the Conda Environment:
   ```
   conda env create -f environment.yml
   
4. Activate the Environment:
    ```
   conda activate nyc_energy
# Usage

## Data Preprocessing

Run the preprocessing script to merge and clean the raw data:

    python src/data_preprocessing.py

This will create a processed file at data/processed/merged_2021.geojson.

## Feature Engineering

Process the training data (2021) to generate features:

    python src/feature_engineering.py

This script reads the processed data, computes derived features (building age, energy intensity, etc.), and integrates weather data.

## Model Training

Train the model using the processed training data:

    python src/train_model.py

This script trains a Random Forest regressor, evaluates performance on a validation set, and saves the model to models/energy_model_rf.pkl.

## Model Evaluation

After processing your testing data (e.g., for 2022) using a similar feature engineering pipeline, run your evaluation script to generate predictions and evaluate model performance.
Data Sources

    Building Footprints: NYC Building Footprints CSV (Building_Footprints_20250211.csv).
    Energy Usage Training: Filtered Energy Usage CSV for 2021 (filtered_evt_EUI-2021.csv).
    Energy Usage Testing: Filtered Energy Usage CSV for 2022 (filtered_evt_EUI-2022.csv).
    Weather Data: Hourly and monthly weather data for NYC (e.g., NYC_Weather_2021.csv and NYC_Weather_2021_Monthly.csv).

# Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

# License

This project is licensed under the MIT License.