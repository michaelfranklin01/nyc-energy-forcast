import os
import pandas as pd
import geopandas as gpd


def get_avg_temp_for_year():
    # Path to the monthly weather CSV for 2021 (already averaged per month)
    weather_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'NYC_Weather_2021_Monthly.csv')
    # Read the monthly weather data; assume it has columns "month" and "avg_temp"
    df_weather = pd.read_csv(weather_path)
    # Compute the overall average temperature for 2021 by averaging the monthly averages
    overall_avg = df_weather['avg_temp'].mean()
    return overall_avg


def create_features():
    # Construct path to processed energy data (merged_2021.geojson)
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'merged_2021.geojson')
    # Read the geojson file using GeoPandas
    gdf = gpd.read_file(data_path)
    # (Optional) If you don't need the geometry for feature engineering, drop it:
    df = pd.DataFrame(gdf.drop(columns='geometry'))
    # Alternatively, if you want to compute spatial features (like area), you can do that here.

    # Calculate building age
    df['building_age'] = 2021 - df['CNSTRCT_YR']

    # Calculate energy intensity (energy per gross floor area)
    df['energy_intensity'] = df.apply(lambda row: row['energy'] / row['gfa'] if row['gfa'] > 0 else None, axis=1)

    # One-hot encode building type
    df = pd.get_dummies(df, columns=['bldgtype'], prefix='type', drop_first=True)

    # Optional: Create age buckets and one-hot encode them
    df['age_bucket'] = pd.cut(df['building_age'], bins=[0, 20, 50, 100, 200],
                              labels=["0-20", "21-50", "51-100", "101+"])
    df = pd.get_dummies(df, columns=['age_bucket'], prefix='age')

    # Incorporate weather data: use the monthly CSV to get the overall average temperature for 2021
    avg_temp = get_avg_temp_for_year()
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
