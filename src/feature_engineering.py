# feature_engineering.py

import os
import pandas as pd
import geopandas as gpd
import datetime

def get_avg_temp_for_year(weather_csv_filename: str) -> float:
    """
    Read monthly-averaged weather CSV and return the overall average temperature.
    """
    weather_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'data', 'raw',
        weather_csv_filename
    )
    df_weather = pd.read_csv(weather_path)
    return df_weather['avg_temp'].mean()

def create_features(
    merged_geojson_filename: str,
    weather_csv_filename: str
) -> (pd.DataFrame, pd.Series):
    """
    Load merged building+energy GeoJSON, engineer features and return (X, y).

    - building_age
    - energy_intensity
    - one-hot bldgtype
    - age_bucket dummies
    - avg_temp (constant)
    """
    # 1) load geometry+properties
    proc_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'data', 'processed',
        merged_geojson_filename
    )
    gdf = gpd.read_file(proc_path)
    df = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))

    # 2) building age
    current_year = datetime.datetime.now().year
    df['building_age'] = current_year - df['CNSTRCT_YR']

    # 3) energy intensity
    df['energy_intensity'] = df.apply(
        lambda r: r['energy'] / r['gfa'] if r['gfa'] > 0 else pd.NA,
        axis=1
    )

    # 4) one-hot bldgtype
    df = pd.get_dummies(df, columns=['bldgtype'], prefix='type', drop_first=True)

    # 5) age buckets
    df['age_bucket'] = pd.cut(
        df['building_age'],
        bins=[0,20,50,100,200],
        labels=['0-20','21-50','51-100','101+']
    )
    df = pd.get_dummies(df, columns=['age_bucket'], prefix='age')

    # 6) avg annual temperature
    df['avg_temp'] = get_avg_temp_for_year(weather_csv_filename)

    # 7) select features + target
    feature_cols = [
        'building_age',
        'gfa',
        'numfloors',
        'energy_intensity',
        'avg_temp'
    ]
    feature_cols += [c for c in df.columns if c.startswith('type_')]
    feature_cols += [c for c in df.columns if c.startswith('age_')]
    target_col = 'energy'

    # 8) drop rows missing any
    df = df.dropna(subset=feature_cols + [target_col])
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    return X, y

if __name__ == "__main__":
    X, y = create_features("merged_2021.geojson", "NYC_weather_2021_monthly.csv")
    print("X shape:", X.shape)
    print(X.head())
    print("y stats:", y.describe())
