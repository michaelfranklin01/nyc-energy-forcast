import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = "filtered_evt.csv"
df = pd.read_csv(file_path)

# Drop rows with missing values in relevant columns
df = df.dropna(subset=['gfa', 'eui', 'primary_type'])

# Global Scatterplot
# Function to create scatter plots
def plot_scatter(x, y, xlabel, ylabel, title):
    #df_filtered = df[df['primary_type'] != 'Multifamily Housing']  # Exclude Multifamily Housing
    plt.figure(figsize=(10, 6))
    #sns.scatterplot(data=df_filtered, x=x, y=y, hue='primary_type', alpha=0.6)
    sns.scatterplot(data=df, x=x, y=y, hue='primary_type', alpha=0.6)
    #plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="Property Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

# GFA Scatter Plots
plot_scatter('gfa', 'eui', "Gross Floor Area (GFA)", "Energy Use Intensity (EUI)", "GFA vs EUI by Property Type")
plot_scatter('gfa', 'wui', "Gross Floor Area (GFA)", "Water Use Intensity (WUI)", "GFA vs WUI by Property Type")
plot_scatter('gfa', 'ghg', "Gross Floor Area (GFA)", "Greenhouse Gas Emissions (GHG)", "GFA vs GHG by Property Type")

# Number of Floors Scatter Plots
plot_scatter('numfloors', 'eui', "Number of Floors", "Energy Use Intensity (EUI)", "Number of Floors vs EUI by Property Type")
plot_scatter('numfloors', 'wui', "Number of Floors", "Water Use Intensity (WUI)", "Number of Floors vs WUI by Property Type")
plot_scatter('numfloors', 'ghg', "Number of Floors", "Greenhouse Gas Emissions (GHG)", "Number of Floors vs GHG by Property Type")

# Number of Buildings Scatter Plots
plot_scatter('numbldgs', 'eui', "Number of Buildings", "Energy Use Intensity (EUI)", "Number of Buildings vs EUI by Property Type")
plot_scatter('numbldgs', 'wui', "Number of Buildings", "Water Use Intensity (WUI)", "Number of Buildings vs WUI by Property Type")
plot_scatter('numbldgs', 'ghg', "Number of Buildings", "Greenhouse Gas Emissions (GHG)", "Number of Buildings vs GHG by Property Type")

# Year Built Scatter Plots
plot_scatter('year', 'eui', "Year Built", "Energy Use Intensity (EUI)", "Year Built vs EUI by Property Type")
plot_scatter('year', 'wui', "Year Built", "Water Use Intensity (WUI)", "Year Built vs WUI by Property Type")
plot_scatter('year', 'ghg', "Year Built", "Greenhouse Gas Emissions (GHG)", "Year Built vs GHG by Property Type")

# Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_matrix = df[['gfa', 'eui', 'wui', 'ghg', 'numfloors', 'numbldgs', 'res_units', 'year']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Box Plots for EUI, WUI, and GHG by Property Type
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='primary_type', y='eui')
plt.xticks(rotation=90)
plt.title("EUI Distribution by Property Type")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='primary_type', y='wui')
plt.xticks(rotation=90)
plt.title("WUI Distribution by Property Type")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='primary_type', y='ghg')
plt.xticks(rotation=90)
plt.title("GHG Distribution by Property Type")
plt.show()

# Violin Plots for EUI, WUI, and GHG by Property Type
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='primary_type', y='eui')
plt.xticks(rotation=90)
plt.title("EUI Violin Plot by Property Type")
plt.show()

plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='primary_type', y='wui')
plt.xticks(rotation=90)
plt.title("WUI Violin Plot by Property Type")
plt.show()

plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='primary_type', y='ghg')
plt.xticks(rotation=90)
plt.title("GHG Violin Plot by Property Type")
plt.show()
