import os
import pandas as pd
import geopandas as gpd


def load_and_clean_data():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # paths to raw data
    path_energy_2021 = os.path.join(script_dir, '..', 'data', 'raw', 'filtered_evt_EUI-2021.csv')
    path_buildings = os.path.join(script_dir, '..', 'data', 'raw', 'Building_Footprints_20250211.csv')

    # Read data
    df_energy_2021 = pd.read_csv(path_energy_2021)
    gdf_buildings = gpd.read_file(path_buildings)

    # Clean / rename columns for consistent merges
    gdf_buildings.rename(columns={'BIN': 'bin'}, inplace=True)
    df_energy_2021.rename(columns={'BIN': 'bin'}, inplace=True)

    # Merge data on bin
    gdf_merged_2021 = gdf_buildings.merge(df_energy_2021, on='bin', how='left')

    # Basic cleaning (drop duplicates, handle missing, etc.)
    gdf_merged_2021.drop_duplicates(subset=['bin'], inplace=True)
    gdf_merged_2021.dropna(subset=['energy'], inplace=True)  # Example

    # Save processed data
    out_path_2021 = os.path.join(script_dir, '..', 'data', 'processed', 'merged_2021.geojson')
    gdf_merged_2021 = gpd.GeoDataFrame(gdf_merged_2021, geometry='the_geom')

    # Write the GeoDataFrame to a GeoJSON file
    gdf_merged_2021.to_file(out_path_2021, driver='GeoJSON')

    return gdf_merged_2021


if __name__ == "__main__":
    # Run the function if we call this script directly
    merged_2021 = load_and_clean_data()
    print("Data loaded and cleaned. Sample:\n", merged_2021.head())
