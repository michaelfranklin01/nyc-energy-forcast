import os
import pandas as pd
import geopandas as gpd
from shapely import wkt


def load_and_clean_data(out_file_name, csv_file):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    csv_filename = f"{csv_file}"
    # Paths to raw data
    path_energy = os.path.join(script_dir, '..', 'data', 'raw', csv_filename)
    path_buildings = os.path.join(script_dir, '..', 'data', 'raw', 'Building_Footprints_20250211.csv')

    # Read the energy CSV and drop its 'the_geom' column if present
    df_energy_2021 = pd.read_csv(path_energy)
    if 'the_geom' in df_energy_2021.columns:
        df_energy_2021.drop(columns=['the_geom'], inplace=True)

    # Read the building footprints CSV and parse geometry from 'the_geom'
    df_buildings = pd.read_csv(path_buildings, low_memory=False)
    # Parse the WKT strings to create valid geometry objects
    df_buildings["geometry"] = df_buildings["the_geom"].apply(wkt.loads)
    # Convert to a GeoDataFrame using the valid geometry column
    gdf_buildings = gpd.GeoDataFrame(
        df_buildings,
        geometry="geometry",
        crs="EPSG:2263"  # NY long island area
    )

    # Rename columns for consistent merging
    gdf_buildings.rename(columns={'BIN': 'bin'}, inplace=True)
    df_energy_2021.rename(columns={'BIN': 'bin'}, inplace=True)
    # Ensure 'bin' is the same type in both datasets
    gdf_buildings['bin'] = gdf_buildings['bin'].astype(str)
    df_energy_2021['bin'] = df_energy_2021['bin'].astype(str)

    # Merge on 'bin'; now there will be no conflicting geometry columns from the energy CSV
    gdf_merged = gdf_buildings.merge(df_energy_2021, on='bin', how='left')

    # Drop duplicates and rows missing critical values
    gdf_merged.drop_duplicates(subset=['bin'], inplace=True)
    gdf_merged.dropna(subset=['energy'], inplace=True)

    # The resulting GeoDataFrame should already have a valid "geometry" field from the building footprints.
    # If any redundant geometry columns exist (e.g., leftover from the merge) drop them:
    for col in ['the_geom', 'the_geom_x', 'the_geom_y']:
        if col in gdf_merged.columns:
            gdf_merged.drop(columns=[col], inplace=True)

    geojson_filename = f"{out_file_name}.geojson"
    # Save processed data to GeoJSON
    out_path = os.path.join(script_dir, '..', 'data', 'processed', geojson_filename)
    gdf_merged.to_file(out_path, driver='GeoJSON')

    return gdf_merged


if __name__ == "__main__":
    merged_2021 = load_and_clean_data("merged_2021.geojson", "filtered_evt_EUI-2021.csv")
    print("Data loaded and cleaned. Sample:\n", merged_2021.head())
