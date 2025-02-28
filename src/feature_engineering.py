import os
import pandas as pd
import geopandas as gpd
import datetime


def get_avg_temp_for_year(weather_data_monthly):
    monthly_weather_data_filename = f"{weather_data_monthly}"
    # Path to the monthly weather CSV for 2021 (already averaged per month)
    weather_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', monthly_weather_data_filename)
    # Read the monthly weather data
    df_weather = pd.read_csv(weather_path)
    # Compute the overall average temperature for 2021 by averaging the monthly averages
    overall_avg = df_weather['avg_temp'].mean()
    return overall_avg


def create_features(training_merged_data, testing_merged_data, weather_data_monthly):
    data_filename = f"{merged_data}"
    # Construct path to processed energy data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', data_filename)
    # Read the geojson file using GeoPandas
    gdf = gpd.read_file(data_path)
    # (Optional) If we don't need the geometry for feature engineering, drop it:
    df = pd.DataFrame(gdf.drop(columns='geometry'))

    # Calculate building age
    current_year = datetime.datetime.now().year
    df['building_age'] = current_year - df['CNSTRCT_YR']

    # Calculate energy intensity (energy per gross floor area)
    df['energy_intensity'] = df.apply(lambda row: row['energy'] / row['gfa'] if row['gfa'] > 0 else None, axis=1)

    # One-hot encode building type
    df = pd.get_dummies(df, columns=['bldgtype'], prefix='type', drop_first=True)

    # Optional: Create age buckets and one-hot encode them
    df['age_bucket'] = pd.cut(df['building_age'], bins=[0, 20, 50, 100, 200],
                              labels=["0-20", "21-50", "51-100", "101+"])
    df = pd.get_dummies(df, columns=['age_bucket'], prefix='age')

    # Incorporate weather data: use the monthly CSV to get the overall average temperature for 2021
    avg_temp = get_avg_temp_for_year(weather_data_monthly)
    df['avg_temp'] = avg_temp

    # Define feature columns and target
    feature_cols = ['building_age', 'gfa', 'numfloors', 'energy_intensity', 'avg_temp']
    feature_cols += [col for col in df.columns if col.startswith('type_')]
    feature_cols += [col for col in df.columns if col.startswith('age_')]
    target_col = 'energy'

    # Drop rows with missing critical features or target
    df = df.dropna(subset=feature_cols + [target_col])

    return df[feature_cols], df[target_col]


if __name__ == "__main__":
    X, y = create_features()
    print("Features sample:")
    print(X.head())
    print("Target sample:")
    print(y.head())
