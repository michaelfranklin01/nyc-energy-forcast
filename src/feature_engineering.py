import os
import pandas as pd


def create_features():
    # Construct path to processed energy data (update file names as needed)
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'merged_2021.csv')
    df = pd.read_csv(data_path)

    # Example feature: building age (assuming 'CNSTRCT_YR' exists)
    df['building_age'] = 2021 - df['CNSTRCT_YR']

    # One-hot encode building type (if 'bldgtype' column exists)
    df = pd.get_dummies(df, columns=['bldgtype'], prefix='type', drop_first=True)

    # Select features and target
    feature_cols = ['building_age', 'gfa', 'numfloors']  # plus any additional features you have
    # Add one-hot encoded building types if available:
    feature_cols += [col for col in df.columns if col.startswith('type_')]

    # Assume target is energy consumption (adjust column name accordingly)
    target_col = 'energy'

    return df[feature_cols], df[target_col]


if __name__ == "__main__":
    X, y = create_features()
    print("Features sample:")
    print(X.head())
    print("Target sample:")
    print(y.head())
