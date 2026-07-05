"""
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
    return trainer