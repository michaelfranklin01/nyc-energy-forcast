# data_analysis.py

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import umap
import shap
from sklearn.inspection import PartialDependenceDisplay
from feature_engineering import create_features
from data_processing import (
    compute_vif,
    pca_explained_variance,
    umap_embedding
)


def visualize_missing(X, y, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    df = X.copy()
    df['energy'] = y
    
    # Set a larger figure size
    plt.figure(figsize=(12, 8))
    msno.matrix(df)
    
    # Save without tight_layout
    plt.savefig(f'{output_dir}/missing_matrix.png', bbox_inches='tight')
    plt.close()

def correlation_map(X, y, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    df = X.copy(); df['energy'] = y
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Map')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/corr_map.png')
    plt.close()

def distribution_plots(X, y, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    df = X.copy(); df['energy'] = y
    for col in df.select_dtypes(include=[np.number]).columns:
        # Skip columns with zero variance
        if df[col].var() == 0:
            print(f"Skipping {col} (zero variance)")
            continue
            
        plt.figure(figsize=(8,4))
        sns.kdeplot(df[col], fill=True)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{col}_dist.png')
        plt.close()

def vif_plot(X, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    vif = compute_vif(X)
    plt.figure(figsize=(8,4))
    sns.barplot(x='VIF', y='feature', data=vif.sort_values('VIF', ascending=False))
    plt.title('VIF per Feature')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/vif.png')
    plt.close()

def pca_plot(X, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    pca, cumvar, n_comp = pca_explained_variance(X)
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1,len(cumvar)+1), cumvar, marker='o')
    plt.axhline(0.90, color='r', linestyle='--')
    plt.title('Cumulative explained variance by PCA')
    plt.xlabel('n components')
    plt.ylabel('cumulative variance')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pca_variance.png')
    plt.close()
    print(f"Components for 90% variance: {n_comp}")

def umap_plot(X, y, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    emb = umap_embedding(X)
    plt.figure(figsize=(6,6))
    sc = plt.scatter(emb[:,0], emb[:,1], c=np.log1p(y), s=3, cmap='Spectral', alpha=0.6)
    plt.colorbar(sc, label='log(energy)')
    plt.title('UMAP embedding')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/umap.png')
    plt.close()

def umap_embedding(X: pd.DataFrame, n_neighbors=50, min_dist=0.3):
    """
    Returns 2D UMAP embedding of X.
    """
    # Add verbose=False to suppress some warnings
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        verbose=False
    )
    embedding = reducer.fit_transform(X)
    return embedding

def shap_summary(model, X, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)

    # Extract inner regressor from TransformedTargetRegressor wrapper
    # model.named_steps['regressor'] is your TransformedTargetRegressor
    # .regressor_ is the fitted underlying regressor
    regressor = model.named_steps['regressor'].regressor_

    explainer = shap.Explainer(regressor)
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_summary.png')
    plt.close()


def pdp_plot(model, X, features, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)

    # Extract inner regressor to avoid issues with pipeline wrapper
    regressor = model.named_steps['regressor'].regressor_

    fig, ax = plt.subplots(figsize=(6,4))
    PartialDependenceDisplay.from_estimator(
        regressor,
        X,
        features,
        ax=ax,
        kind='average'
    )
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pdp_{"_".join(features)}.png')
    plt.close()


if __name__ == "__main__":
    merged = "merged_2021.geojson"
    weather = "NYC_weather_2021_monthly.csv"
    X, y = create_features(merged, weather)
    print("Running EDA on", X.shape, "samples")

    visualize_missing(X, y)
    correlation_map(X, y)
    distribution_plots(X, y)
    vif_plot(X)
    pca_plot(X)
    umap_plot(X, y)

    # Check if model file exists
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'energy_rf.pkl')
    if os.path.exists(model_path):
        from joblib import load
        try:
            model = load(model_path)
            shap_summary(model, X)
            pdp_plot(model, X, ['gfa','building_age'])
            print("Model analysis completed successfully")
        except Exception as e:
            print(f"Error analyzing model: {e}")
    else:
        print(f"Model file not found: {model_path}")
        print("Skipping model analysis steps")
