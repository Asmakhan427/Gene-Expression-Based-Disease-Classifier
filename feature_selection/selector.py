"""
selector.py - Feature selection pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from utils.logger import setup_logger

logger = setup_logger(__name__)


def run_feature_selection(X_train, y_train, config):
    """
    Run feature selection on training data.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        config: Configuration dict with feature_selection settings
    
    Returns:
        selected_features: List of selected feature names
        scores: Feature importance scores
    """
    fs_config = config.get("feature_selection", {})
    method = fs_config.get("method", "selectkbest")
    n_features = fs_config.get("n_features", min(20, X_train.shape[1]))
    
    logger.info(f"Running {method} feature selection...")
    
    if method == "selectkbest":
        score_func = fs_config.get("score_func", "f_classif")
        if score_func == "mutual_info":
            score_fn = mutual_info_classif
        else:
            score_fn = f_classif
        
        selector = SelectKBest(score_function=score_fn, k=n_features)
        selector.fit(X_train, y_train)
        
        selected_mask = selector.get_support()
        selected_features = X_train.columns[selected_mask].tolist()
        scores = selector.scores_
        
        logger.info(f"Selected {len(selected_features)} features")
        
        return selected_features, scores
    else:
        logger.warning(f"Unknown method {method}, returning all features")
        return list(X_train.columns), np.ones(X_train.shape[1])


def load_feature_selection(filepath):
    """Load pre-computed feature selection results."""
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)