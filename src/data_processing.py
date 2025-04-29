# data_processing.py

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_log_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
import umap

def make_log_regressor(regressor):
    """
    Wrap any regressor to learn log1p(target) and invert back.
    """
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1
    )

def compute_rmsle(y_true, y_pred):
    """
    Root Mean Squared Log Error.
    """
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Returns VIF DataFrame for each numeric column in X.
    """
    X_num = X.select_dtypes(include=[np.number]).dropna()
    vif_data = pd.DataFrame({
        'feature': X_num.columns,
        'VIF': [
            variance_inflation_factor(X_num.values, i)
            for i in range(X_num.shape[1])
        ]
    })
    return vif_data

def pca_explained_variance(X: pd.DataFrame, threshold: float = 0.90):
    """
    Fit PCA, plot cumulative explained variance, and return n_components
    to reach threshold.
    """
    pca = PCA()
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cumvar, threshold) + 1)
    return pca, cumvar, n_comp

def umap_embedding(X: pd.DataFrame, n_neighbors=50, min_dist=0.3):
    """
    Returns 2D UMAP embedding of X.
    """
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding = reducer.fit_transform(X)
    return embedding

def lasso_feature_selection(X, y, cv=5):
    """
    LassoCV to select features with non-zero coefficients.
    """
    from sklearn.linear_model import LassoCV
    lasso = LassoCV(cv=cv, random_state=42).fit(X, y)
    coefs = pd.Series(lasso.coef_, index=X.columns)
    selected = coefs[coefs.abs()>1e-4].index.tolist()
    return selected, coefs

def rfe_feature_selection(X, y, n_features_to_select=20, step=5):
    """
    Recursive Feature Elimination with RF base to choose top features.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import RFE
    base = RandomForestRegressor(n_estimators=100, random_state=42)
    selector = RFE(base, n_features_to_select=n_features_to_select, step=step).fit(X, y)
    selected = X.columns[selector.support_].tolist()
    return selected, selector.ranking_

# Advanced CV splitters
def get_group_kfold(n_splits=5):
    from sklearn.model_selection import GroupKFold
    return GroupKFold(n_splits=n_splits)

def get_time_series_split(n_splits=5):
    from sklearn.model_selection import TimeSeriesSplit
    return TimeSeriesSplit(n_splits=n_splits)
