"""
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
    
    logger.info("EDA complete. Visualizations saved.")