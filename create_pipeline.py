#!/usr/bin/env python3
"""
COMPLETE FIX SCRIPT
Fixes null bytes AND creates all missing modules
Run this to get a fully working pipeline
"""

from pathlib import Path
import shutil
from datetime import datetime

# All required Python files with clean content
REQUIRED_FILES = {
    # Utils
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

    # Data
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

    # Preprocessing
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
    return X_train, X_test, y_train, y_test, scaler, list(X_norm.columns)


def load_processed(filepath):
    """Load pre-processed data."""
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)''',

    # Feature Selection
    "feature_selection/__init__.py": '''"""
Feature selection module.
"""
from .selector import run_feature_selection

__all__ = ['run_feature_selection']''',

    "feature_selection/selector.py": '''"""
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
        return pickle.load(f)''',

    # ML Models
    "ml/__init__.py": '''"""
Machine learning module.
"""
from .models import run_ml_pipeline

__all__ = ['run_ml_pipeline']''',

    "ml/models.py": '''"""
models.py - Machine learning models and training pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

from utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelTrainer:
    """Train and compare multiple ML models."""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all configured models."""
        ml_config = self.config.get("ml", {})
        
        models_to_train = {
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ),
            "svm": SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),
        }
        
        logger.info("Training multiple models...")
        
        for model_name, model in models_to_train.items():
            try:
                logger.info(f"  Training {model_name}...")
                model.fit(X_train, y_train)
                self.models[model_name] = model
                
                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    roc_auc = roc_auc_score(y_test, y_proba)
                except:
                    roc_auc = None
                
                self.results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc,
                }
                
                logger.info(f"    Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
        
        # Select best model by F1 score
        best_f1 = -1
        for model_name, metrics in self.results.items():
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                self.best_model_name = model_name
                self.best_model = self.models[model_name]
        
        logger.info(f"Best model: {self.best_model_name} (F1: {best_f1:.4f})")
        
        return self.best_model
    
    def get_model_comparison_df(self):
        """Return comparison dataframe of all models."""
        df = pd.DataFrame(self.results).T
        df = df.round(4)
        return df


def run_ml_pipeline(X_train, y_train, X_test, y_test, config):
    """
    Run the machine learning pipeline.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        config: Configuration dict
    
    Returns:
        trainer: Trained ModelTrainer instance
    """
    trainer = ModelTrainer(config)
    trainer.train_models(X_train, y_train, X_test, y_test)
    return trainer''',

    # Biomarker Analysis
    "biomarker_analysis/__init__.py": '''"""
Biomarker analysis module.
"""
from .analyzer import run_biomarker_analysis

__all__ = ['run_biomarker_analysis']''',

    "biomarker_analysis/analyzer.py": '''"""
analyzer.py - Biomarker discovery and analysis.
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.preprocessing import StandardScaler

from utils.logger import setup_logger

logger = setup_logger(__name__)


class BiomarkerAnalyzer:
    """Identify and analyze potential disease biomarkers."""
    
    def __init__(self, config):
        self.config = config
        self.biomarkers = None
        self.p_values = None
        self.fold_changes = None
    
    def find_biomarkers(self, X_train, y_train, X_test, y_test):
        """
        Identify biomarkers using statistical tests.
        """
        logger.info("Analyzing potential biomarkers...")
        
        # Separate by class
        classes = np.unique(y_train)
        
        if len(classes) == 2:
            class_0_mask = y_train == classes[0]
            class_1_mask = y_train == classes[1]
            
            X_class_0 = X_train[class_0_mask]
            X_class_1 = X_train[class_1_mask]
            
            # Calculate statistics for each feature
            p_values = []
            fold_changes = []
            
            for col in X_train.columns:
                # T-test
                try:
                    _, p_val = ttest_ind(X_class_0[col], X_class_1[col])
                    p_values.append(p_val)
                except:
                    p_values.append(1.0)
                
                # Fold change
                mean_0 = X_class_0[col].mean()
                mean_1 = X_class_1[col].mean()
                if mean_0 > 0:
                    fc = mean_1 / mean_0
                else:
                    fc = 0
                fold_changes.append(fc)
            
            self.p_values = np.array(p_values)
            self.fold_changes = np.array(fold_changes)
            
            # Select biomarkers (p < 0.05 and |log2(FC)| > 1)
            threshold = self.config.get("biomarker", {}).get("p_threshold", 0.05)
            fc_threshold = self.config.get("biomarker", {}).get("fc_threshold", 1.0)
            
            significant = (self.p_values < threshold) & (np.abs(np.log2(np.array(fold_changes) + 1e-6)) > fc_threshold)
            self.biomarkers = X_train.columns[significant].tolist()
            
            logger.info(f"Found {len(self.biomarkers)} significant biomarkers")
            
            return self.biomarkers
        else:
            logger.warning("Multi-class classification not yet supported for biomarker analysis")
            return []
    
    def get_biomarker_info(self):
        """Get detailed biomarker information."""
        if self.biomarkers is None:
            return None
        
        biomarker_data = []
        for i, biomarker in enumerate(self.biomarkers):
            # Find index in original features
            idx = list(self.biomarkers).index(biomarker)
            biomarker_data.append({
                'Biomarker': biomarker,
                'P-Value': self.p_values[idx],
                'Fold-Change': self.fold_changes[idx],
            })
        
        return pd.DataFrame(biomarker_data).sort_values('P-Value')


def run_biomarker_analysis(X_train, y_train, X_test, y_test, config):
    """
    Run biomarker discovery analysis.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        config: Configuration dict
    
    Returns:
        analyzer: BiomarkerAnalyzer instance
    """
    analyzer = BiomarkerAnalyzer(config)
    analyzer.find_biomarkers(X_train, y_train, X_test, y_test)
    return analyzer''',

    # Explainability
    "explainability/__init__.py": '''"""
Explainability module.
"""
from .shap_analyzer import run_shap_analysis

__all__ = ['run_shap_analysis']''',

    "explainability/shap_analyzer.py": '''"""
shap_analyzer.py - SHAP-based explainability analysis.
"""

import numpy as np
import pandas as pd

from utils.logger import setup_logger

logger = setup_logger(__name__)


class SHAPAnalyzer:
    """SHAP-based model explainability."""
    
    def __init__(self, model, X_train, X_test, config):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.config = config
        self.shap_values = None
        self.feature_importance = None
    
    def analyze(self):
        """
        Analyze model using SHAP values.
        Falls back to permutation importance if SHAP unavailable.
        """
        try:
            import shap
            logger.info("Computing SHAP values...")
            
            # Use KernelExplainer for model-agnostic approach
            explainer = shap.KernelExplainer(
                self.model.predict_proba,
                shap.sample(self.X_train, min(100, len(self.X_train)))
            )
            self.shap_values = explainer.shap_values(self.X_test)
            
            # Get feature importance from mean absolute SHAP values
            if isinstance(self.shap_values, list):
                # Multi-class
                shap_vals = np.abs(self.shap_values[1])
            else:
                shap_vals = np.abs(self.shap_values)
            
            feature_importance = np.mean(shap_vals, axis=0)
            
        except (ImportError, Exception) as e:
            logger.warning(f"SHAP analysis failed: {e}. Using permutation importance instead.")
            feature_importance = self._permutation_importance()
        
        self.feature_importance = feature_importance
        return self
    
    def _permutation_importance(self):
        """Fallback to permutation importance."""
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(
            self.model,
            self.X_test,
            np.arange(len(self.X_test)),
            n_repeats=10,
            random_state=42
        )
        
        return result.importances_mean
    
    def get_feature_importance_df(self):
        """Get feature importance as dataframe."""
        if self.feature_importance is None:
            return None
        
        importance_df = pd.DataFrame({
            'Feature': self.X_test.columns,
            'Importance': self.feature_importance,
        }).sort_values('Importance', ascending=False)
        
        return importance_df


def run_shap_analysis(model, X_train, X_test, config):
    """
    Run SHAP explainability analysis.
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        config: Configuration dict
    
    Returns:
        analyzer: SHAPAnalyzer instance
    """
    analyzer = SHAPAnalyzer(model, X_train, X_test, config)
    analyzer.analyze()
    return analyzer''',

    # Visualization
    "visualization/__init__.py": '''"""
Visualization module.
"""
from .visualizer import run_eda

__all__ = ['run_eda']''',

    "visualization/visualizer.py": '''"""
visualizer.py - Visualization module for EDA and results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.logger import setup_logger

logger = setup_logger(__name__)


def run_eda(X, y, config):
    """
    Perform exploratory data analysis with visualizations.
    
    Args:
        X: Feature matrix
        y: Labels
        config: Configuration dict
    """
    logger.info("Running exploratory data analysis...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Class distribution
    try:
        plt.figure(figsize=(10, 5))
        sns.countplot(x=y)
        plt.title("Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("eda_class_distribution.png", dpi=100, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: eda_class_distribution.png")
    except Exception as e:
        logger.warning(f"Could not create class distribution plot: {e}")
    
    # Feature statistics
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Feature means
        feature_means = X.mean()
        feature_means.plot(kind='barh', ax=axes[0])
        axes[0].set_title("Mean Feature Expression")
        axes[0].set_xlabel("Mean Value")
        
        # Feature variance
        feature_var = X.var()
        feature_var.plot(kind='barh', ax=axes[1])
        axes[1].set_title("Feature Variance")
        axes[1].set_xlabel("Variance")
        
        plt.tight_layout()
        plt.savefig("eda_feature_stats.png", dpi=100, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: eda_feature_stats.png")
    except Exception as e:
        logger.warning(f"Could not create feature stats plot: {e}")
    
    # PCA visualization (if features > 2)
    try:
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            plt.figure(figsize=(10, 7))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
            plt.title("PCA Visualization")
            plt.colorbar(scatter, label="Class")
            plt.tight_layout()
            plt.savefig("eda_pca.png", dpi=100, bbox_inches='tight')
            plt.close()
            logger.info("✓ Saved: eda_pca.png")
    except Exception as e:
        logger.warning(f"Could not create PCA plot: {e}")
    
    # Heatmap of top features
    try:
        n_features_to_show = min(20, X.shape[1])
        X_top = X.iloc[:, :n_features_to_show]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(X_top.head(50), cmap='coolwarm', xticklabels=True)
        plt.title("Top Features Heatmap (First 50 Samples)")
        plt.tight_layout()
        plt.savefig("eda_heatmap.png", dpi=100, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: eda_heatmap.png")
    except Exception as e:
        logger.warning(f"Could not create heatmap: {e}")
    
    logger.info("EDA complete. Visualizations saved.")''',
}


def main():
    """Create all required project files with clean content."""
    project_root = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = project_root / f"_backup_complete_{timestamp}"
    
    print("\n" + "=" * 80)
    print("🔧 COMPLETE PROJECT FIX - Creating/Fixing All Modules")
    print("=" * 80)
    
    # Create backup directory
    backup_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📦 Backing up existing files to: {backup_dir}\n")
    
    created = 0
    replaced = 0
    
    for filepath, content in REQUIRED_FILES.items():
        full_path = project_root / filepath
        
        # Create parent directories
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Backup if exists
        if full_path.exists():
            backup_path = backup_dir / filepath.replace("/", "_")
            shutil.copy2(full_path, backup_path)
            replaced += 1
            print(f"🔄 Replaced: {filepath}")
        else:
            created += 1
            print(f"✨ Created: {filepath}")
        
        # Write clean content
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print("\n" + "=" * 80)
    print(f"✅ SUCCESS!")
    print(f"   • Created {created} new files")
    print(f"   • Fixed {replaced} existing files")
    print(f"📌 Backup: {backup_dir}")
    print("=" * 80)
    print("\n🚀 Next steps:")
    print("   1. python pipeline.py        # Run the pipeline")
    print("   2. python streamlit_app.py   # Or run the Streamlit app")
    print("\n")


if __name__ == "__main__":
    main()