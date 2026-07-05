"""
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
    return analyzer