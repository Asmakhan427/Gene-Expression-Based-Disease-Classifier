#!/usr/bin/env python3
"""
Automated fix script to replace corrupted files with clean UTF-8 versions.
Run this script to fix all null byte issues at once.
"""

from pathlib import Path
import shutil
from datetime import datetime

# Clean file contents (UTF-8 encoded)
CLEAN_FILES = {
    "utils/__init__.py": '''"""
Utility modules.
"""
from .config_loader import load_config, ensure_dirs
from .logger import setup_logger

__all__ = [
    'load_config',
    'ensure_dirs',
    'setup_logger',
]''',

    "utils/config_loader.py": '''"""
config_loader.py - Central configuration loader.
"""

from pathlib import Path
from typing import Any
import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration file."""
    config_path = Path(path) if path else _CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(config: dict[str, Any]) -> None:
    """Create all directories defined in config."""
    for key in config.get("paths", {}):
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)''',

    "utils/logger.py": '''"""
logger.py - Centralised logging configuration.
"""

import logging
from pathlib import Path


def setup_logger(
    name: str = "gene_classifier",
    level: str = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure and return a project-wide logger."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(name)
    log.setLevel(numeric_level)
    if not log.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(numeric_level)
        ch.setFormatter(formatter)
        log.addHandler(ch)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(numeric_level)
            fh.setFormatter(formatter)
            log.addHandler(fh)
    return log''',

    "data/__init__.py": '''"""
Data acquisition module.
"""
from .acquire_data import acquire_dataset

__all__ = ['acquire_dataset']''',

    "data/acquire_data.py": '''"""
acquire_data.py - Dataset acquisition module.
"""

from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm

from utils.config_loader import load_config
from utils.logger import setup_logger

logger = setup_logger(__name__)

EXPECTED_FILE = "breast_cancer_gene_expression.csv"
FALLBACK_URL = (
    "https://raw.githubusercontent.com/brunogrisci/"
    "breast-cancer-gene-expression-profiling/master/"
    "breast_cancer_gene_expression.csv"
)


def acquire_dataset(config=None):
    """Download the breast-cancer gene expression dataset."""
    if config is None:
        config = load_config()
    raw_dir = Path(config["paths"]["data_raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_file = raw_dir / EXPECTED_FILE
    if target_file.exists():
        logger.info("Dataset already present at %s", target_file)
        return pd.read_csv(target_file, index_col=0)
    logger.info("Downloading from %s …", FALLBACK_URL)
    response = requests.get(FALLBACK_URL, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with target_file.open("wb") as fh, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            fh.write(chunk)
            bar.update(len(chunk))
    return pd.read_csv(target_file, index_col=0)''',

    "preprocessing/__init__.py": '''"""
Preprocessing module.
"""
from .preprocessor import run_preprocessing

__all__ = ['run_preprocessing']''',

    "preprocessing/preprocessor.py": '''"""
preprocessor.py - Gene-expression preprocessing pipeline.
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
    """Execute the full preprocessing pipeline."""
    if config is None:
        config = load_config()
    pcfg = config["preprocessing"]
    label_col = config["dataset"]["label_column"]
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found.")
    y = df[label_col].copy()
    X = df.drop(columns=[label_col])
    X = X.drop_duplicates()
    missing_frac = X.isnull().mean()
    to_drop = missing_frac[missing_frac > pcfg["missing_threshold"]].index
    X = X.drop(columns=to_drop)
    if X.isnull().any().any():
        X = X.fillna(X.median())
    selector = VarianceThreshold(threshold=pcfg["variance_threshold"])
    arr = selector.fit_transform(X.values)
    selected_cols = X.columns[selector.get_support()].tolist()
    X = pd.DataFrame(arr, index=X.index, columns=selected_cols)
    if pcfg["log_transform"]:
        min_val = X.values.min()
        if min_val < 0:
            X = X - min_val
        X = np.log1p(X)
    scaler = QuantileTransformer(
        n_quantiles=min(X.shape[0], 1000),
        output_distribution="normal",
        random_state=42
    )
    arr = scaler.fit_transform(X.values)
    X_norm = pd.DataFrame(arr, index=X.index, columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm,
        y,
        test_size=pcfg["test_size"],
        random_state=pcfg["random_state"],
        stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler, list(X_norm.columns)''',
}


def main():
    """Replace all potentially corrupted files with clean versions."""
    project_root = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = project_root / f"_backup_{timestamp}"
    
    print("=" * 70)
    print("🔧 FIXING NULL BYTES - AUTOMATED REPAIR")
    print("=" * 70)
    
    # Create backup directory
    backup_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📦 Creating backup in: {backup_dir}\n")
    
    replaced_count = 0
    
    for filepath, content in CLEAN_FILES.items():
        full_path = project_root / filepath
        
        # Create parent directories if they don't exist
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Backup existing file if it exists
        if full_path.exists():
            backup_path = backup_dir / filepath.replace("/", "_")
            shutil.copy2(full_path, backup_path)
            print(f"💾 Backed up: {filepath}")
        
        # Write clean content
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        replaced_count += 1
        print(f"✅ Fixed: {filepath}")
    
    print("\n" + "=" * 70)
    print(f"✨ SUCCESS! Fixed {replaced_count} files")
    print(f"📌 Backup created at: {backup_dir}")
    print("=" * 70)
    print("\n🚀 Your pipeline should now run without null byte errors!")
    print("   Try: python pipeline.py\n")


if __name__ == "__main__":
    main()