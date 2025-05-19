# train_model.py

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from data_processing import (
    make_log_regressor,
    compute_rmsle,
    get_group_kfold,
    get_time_series_split
)

def hyperparameter_search(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series = None,
    use_time_series: bool = False,
    n_iter: int = 20
):
    """
    Randomized search over RF hyperparams, with optional GroupKFold or TimeSeriesSplit.
    """
    cv = (get_time_series_split() if use_time_series
          else get_group_kfold() if groups is not None
          else 5)

    # Update: Removed 'auto' which is deprecated in newer scikit-learn
    param_dist = {
        'regressor__regressor__n_estimators': [100, 200, 500], 
        'regressor__regressor__max_depth': [None, 10, 20, 30],
        'regressor__regressor__min_samples_leaf': [1, 2, 5],
        'regressor__regressor__max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7]  # Updated values
    }

    # wrap in log-regressor
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('regressor', make_log_regressor(RandomForestRegressor(random_state=42)))
    ])

    rnd_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        random_state=42
    )
    rnd_search.fit(X, y, **({'groups': groups} if groups is not None else {}))
    return rnd_search

def train_final_model(X, y, best_params: dict, output_path=None):
    """
    Train a final RandomForest (wrapped in log-transform) using best_params,
    evaluate with CV, and optionally save to disk.
    """
    from sklearn.pipeline import Pipeline
    from data_processing import make_log_regressor

    pipeline = Pipeline([
        ('regressor', make_log_regressor(RandomForestRegressor(**best_params, random_state=42)))
    ])
    # 5-fold CV
    scores = cross_val_score(
        pipeline, X, y,
        scoring=lambda est, X_test, y_test: -compute_rmsle(y_test, est.predict(X_test)),
        cv=5,
        n_jobs=-1
    )
    print("CV RMSLE:", -scores.mean(), "±", scores.std())

    pipeline.fit(X, y)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(pipeline, output_path)
        print("Model saved to:", output_path)
    return pipeline

if __name__ == "__main__":
    from feature_engineering import create_features

    # load
    X, y = create_features("merged_2021.geojson", "NYC_weather_2021_monthly.csv")

    # Example: spatial CV by neighborhood (if you have a 'neighborhood' column)
    groups = X.get('neighborhood', None)

    # 1) hyperparam search
    rnd = hyperparameter_search(X, y, groups=groups, use_time_series=False, n_iter=20)
    print("Best params:", rnd.best_params_)

    # 2) train final model
    model = train_final_model(
        X, y,
    best_params = {
    k.replace('regressor__regressor__', ''): v 
    for k, v in rnd.best_params_.items()
    },
        output_path=os.path.join(os.path.dirname(__file__),'..','models','energy_rf.pkl')
    )
