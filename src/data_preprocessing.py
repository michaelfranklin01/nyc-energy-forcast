import os
import pandas as pd
import geopandas as gpd
from shapely import wkt


def load_and_clean_data():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths to raw data
    path_energy_2021 = os.path.join(script_dir, '..', 'data', 'raw', 'filtered_evt_EUI-2021.csv')
    path_buildings = os.path.join(script_dir, '..', 'data', 'raw', 'Building_Footprints_20250211.csv')

    # Read the energy CSV
    df_energy_2021 = pd.read_csv(path_energy_2021)

    # Read the building footprints CSV (contains WKT in 'the_geom')
    df_buildings = pd.read_csv(path_buildings, low_memory=False)
    # Parse the WKT string into a Shapely geometry
    df_buildings["geometry"] = df_buildings["the_geom"].apply(wkt.loads)

    # Convert to GeoDataFrame, specifying the geometry column and CRS
    gdf_buildings = gpd.GeoDataFrame(
        df_buildings,
        geometry="geometry",
        crs="EPSG:2263"  # NY State Plane (Long Island), or your correct CRS
    )

    # Rename columns for consistent merges
    gdf_buildings.rename(columns={'BIN': 'bin'}, inplace=True)
    df_energy_2021.rename(columns={'BIN': 'bin'}, inplace=True)

    # Ensure 'bin' is the same type in both
    gdf_buildings['bin'] = gdf_buildings['bin'].astype(str)
    df_energy_2021['bin'] = df_energy_2021['bin'].astype(str)

    # Merge data on 'bin'
    gdf_merged_2021 = gdf_buildings.merge(df_energy_2021, on='bin', how='left')

    # Drop duplicates and rows missing 'energy'
    gdf_merged_2021.drop_duplicates(subset=['bin'], inplace=True)
    gdf_merged_2021.dropna(subset=['energy'], inplace=True)

    # Convert merged DataFrame back to a GeoDataFrame,
    # using the parsed 'geometry' column from the footprints
    gdf_merged_2021 = gpd.GeoDataFrame(
        gdf_merged_2021,
        geometry='geometry',
        crs="EPSG:2263"
    )

    # Save the merged GeoDataFrame to GeoJSON
    out_path_2021 = os.path.join(script_dir, '..', 'data', 'processed', 'merged_2021.geojson')
    gdf_merged_2021.to_file(out_path_2021, driver='GeoJSON')

    return gdf_merged_2021


if __name__ == "__main__":
    # Run the function if we call this script directly
    merged_2021 = load_and_clean_data()
    print("Data loaded and cleaned. Sample:\n", merged_2021.head())
