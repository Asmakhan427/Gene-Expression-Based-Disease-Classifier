"""
acquire_data.py - Fixed dataset loading
"""

from pathlib import Path
import pandas as pd
import requests

from utils.config_loader import load_config
from utils.logger import setup_logger

logger = setup_logger(__name__)

EXPECTED_FILE = "breast_cancer_gene_expression.csv"


def acquire_dataset(config=None):
    """Load the breast-cancer gene expression dataset."""
    if config is None:
        config = load_config()
    
    raw_dir = Path(config["paths"]["data_raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_file = raw_dir / EXPECTED_FILE
    
    # Check if already downloaded
    if target_file.exists():
        logger.info(f"✅ Loading dataset from: {target_file}")
        
        # Load the CSV properly
        df = pd.read_csv(target_file, index_col=0)
        
        # CRITICAL FIX: Check if the dataframe is empty
        if len(df) == 0:
            logger.warning("⚠️ Dataset is empty! Re-downloading...")
            target_file.unlink()
            return _download_and_save(raw_dir, target_file, config)
        
        # Ensure 'class' column exists
        if 'class' not in df.columns:
            logger.warning(f"⚠️ 'class' column not found. Columns: {df.columns.tolist()}")
            # Try to find a column that might be the class
            for col in df.columns:
                if 'class' in col.lower() or 'target' in col.lower():
                    df = df.rename(columns={col: 'class'})
                    logger.info(f"✅ Renamed '{col}' to 'class'")
                    break
        
        # Ensure class column has correct values
        if 'class' in df.columns:
            # Convert numeric to string labels if needed
            if df['class'].dtype in ['int64', 'float64']:
                df['class'] = df['class'].map({2: 'normal', 4: 'cancer'})
            
            # Verify we have both classes
            classes = df['class'].unique().tolist()
            logger.info(f"📊 Classes in dataset: {classes}")
            
            if len(classes) != 2:
                logger.warning(f"⚠️ Expected 2 classes, found {len(classes)}: {classes}")
        
        logger.info(f"✅ Dataset loaded: {len(df)} samples × {len(df.columns)} features")
        return df
    
    # If file doesn't exist, download it
    return _download_and_save(raw_dir, target_file, config)


def _download_and_save(raw_dir, target_file, config):
    """Download the dataset from UCI and save it."""
    logger.info("📊 Downloading dataset from UCI...")
    
    try:
        response = requests.get(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
            timeout=30
        )
        response.raise_for_status()
        
        # Parse the data
        lines = response.text.strip().split('\n')
        data = []
        
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 11 and '?' not in parts:
                # Class: 2 = benign (normal), 4 = malignant (cancer)
                class_label = 'normal' if int(parts[10]) == 2 else 'cancer'
                
                sample = {
                    'id': int(parts[0]),
                    'clump_thickness': float(parts[1]),
                    'uniformity_cell_size': float(parts[2]),
                    'uniformity_cell_shape': float(parts[3]),
                    'marginal_adhesion': float(parts[4]),
                    'single_epithelial_cell_size': float(parts[5]),
                    'bare_nuclei': float(parts[6]),
                    'bland_chromatin': float(parts[7]),
                    'normal_nucleoli': float(parts[8]),
                    'mitoses': float(parts[9]),
                    'class': class_label
                }
                data.append(sample)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df = df.set_index('id')
        
        # Clean data
        for col in df.columns:
            if col != 'class':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        
        # Save to file
        df.to_csv(target_file)
        
        logger.info(f"✅ Dataset downloaded and saved: {len(df)} samples")
        logger.info(f"   Classes: {df['class'].unique().tolist()}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        raise RuntimeError(f"Could not download dataset: {e}")
