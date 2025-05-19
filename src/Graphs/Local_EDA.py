import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = "filtered_evt.csv"
df = pd.read_csv(file_path)

# Drop rows with missing values in relevant columns
df = df.dropna(subset=['gfa', 'eui', 'primary_type'])

# Separate scatter plots for each property type
unique_types = df['primary_type'].unique()
def plot_by_property_type(x, y, xlabel, ylabel):
    for prop_type in unique_types:
        subset = df[df['primary_type'] == prop_type]
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=subset, x=x, y=y, alpha=0.6)
        #plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{xlabel} vs {ylabel} for {prop_type}")
        plt.show()

# GFA vs EUI, WUI, GHG for each property type
# plot_by_property_type('gfa', 'eui', "Gross Floor Area (GFA)", "Energy Use Intensity (EUI)")
# plot_by_property_type('gfa', 'wui', "Gross Floor Area (GFA)", "Water Use Intensity (WUI)")
# plot_by_property_type('gfa', 'ghg', "Gross Floor Area (GFA)", "Greenhouse Gas Emissions (GHG)")

# plot_by_property_type('numfloors', 'eui', "Number of Floors", "Energy Use Intensity (EUI)")
# plot_by_property_type('numfloors', 'wui', "Number of Floors", "Water Use Intensity (WUI)")
# plot_by_property_type('numfloors', 'ghg', "Number of Floors", "Greenhouse Gas Emissions (GHG)")

# plot_by_property_type('numbldgs', 'eui', "Number of Buildings", "Energy Use Intensity (EUI)")
# plot_by_property_type('numbldgs', 'wui', "Number of Buildings", "Water Use Intensity (WUI)")
# plot_by_property_type('numbldgs', 'ghg', "Number of Buildings", "Greenhouse Gas Emissions (GHG)")

# plot_by_property_type('year', 'eui', "Year Built", "Energy Use Intensity (EUI)")
# plot_by_property_type('year', 'wui', "Year Built", "Water Use Intensity (WUI)")
# plot_by_property_type('year', 'ghg', "Year Built", "Greenhouse Gas Emissions (GHG)")

# Separate Correlation Heatmaps by Property Type
unique_types = df['primary_type'].unique()
for prop_type in unique_types:
    subset = df[df['primary_type'] == prop_type]
    plt.figure(figsize=(10, 8))
    corr_matrix = subset[['gfa', 'eui', 'wui', 'ghg', 'numfloors', 'numbldgs', 'year']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f"Correlation Heatmap for {prop_type}")
    plt.show()