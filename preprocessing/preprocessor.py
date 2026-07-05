"""
preprocessor.py - Gene-expression preprocessing pipeline with saving
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import VarianceThreshold

from utils.config_loader import load_config
from utils.logger import setup_logger

logger = setup_logger(__name__)


def run_preprocessing(df, config=None):
    """Execute the full preprocessing pipeline and SAVE the data."""
    if config is None:
        config = load_config()
    
    pcfg = config["preprocessing"]
    label_col = config["dataset"]["label_column"]
    
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found.")
    
    # Separate features and labels
    y = df[label_col].copy()
    X = df.drop(columns=[label_col])
    
    # Remove duplicate rows
    if X.index.duplicated().any():
        keep_idx = ~X.index.duplicated()
        X = X[keep_idx]
        y = y[keep_idx]
    
    # Handle missing values - fill with median
    if X.isnull().any().any():
        X = X.fillna(X.median())
    
    # Remove rows with NaN and sync y
    nan_rows = X.isnull().any(axis=1)
    if nan_rows.any():
        keep_idx = ~nan_rows
        X = X.loc[keep_idx]
        y = y.loc[keep_idx]
    
    # Verify shapes match
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows, y has {y.shape[0]} rows!")
    
    # Variance filter
    selector = VarianceThreshold(threshold=pcfg["variance_threshold"])
    arr = selector.fit_transform(X.values)
    selected_cols = X.columns[selector.get_support()].tolist()
    X = pd.DataFrame(arr, index=X.index, columns=selected_cols)
    
    # Log transformation
    if pcfg["log_transform"]:
        min_val = X.values.min()
        if min_val < 0:
            X = X - min_val
        X = np.log1p(X)
    
    # Normalisation
    scaler = QuantileTransformer(
        n_quantiles=min(X.shape[0], 1000),
        output_distribution="normal",
        random_state=42
    )
    arr = scaler.fit_transform(X.values)
    X_norm = pd.DataFrame(arr, index=X.index, columns=X.columns)
    
    # Final verification
    assert X_norm.shape[0] == y.shape[0]
    
    logger.info(f"✅ X shape: {X_norm.shape}, y shape: {y.shape}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y,
        test_size=pcfg["test_size"],
        random_state=pcfg["random_state"],
        stratify=y
    )
    
    # ⭐ CRITICAL: Save processed data as DICT
    processed_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": list(X_norm.columns)
    }
    
    # Save to file
    save_path = Path(config["paths"]["data_processed"]) / "processed_data.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(processed_data, f)
    
    logger.info(f"✅ Processed data SAVED to: {save_path}")
    logger.info(f"✅ Final - X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"✅ Final - X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler, list(X_norm.columns)


def load_processed(filepath=None):
    """
    Load pre-processed data from pickle file.
    
    Returns
    -------
    dict
        Dictionary containing X_train, X_test, y_train, y_test, scaler, feature_names
    """
    if filepath is None:
        config = load_config()
        filepath = Path(config["paths"]["data_processed"]) / "processed_data.pkl"
    else:
        filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Processed data not found at: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"✅ Loaded processed data from: {filepath}")
    return data
