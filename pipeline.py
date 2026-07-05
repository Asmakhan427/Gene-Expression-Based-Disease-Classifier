"""
pipeline.py - Gene Expression Classification Pipeline
"""

from __future__ import annotations

import logging
import sys
import warnings
import pickle
from pathlib import Path
from typing import Any

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from utils.config_loader import load_config, ensure_dirs
from utils.logger import setup_logger
from data.acquire_data import acquire_dataset
from preprocessing.preprocessor import run_preprocessing, load_processed
from feature_selection.selector import run_feature_selection, load_feature_selection
from biomarker_analysis.analyzer import run_biomarker_analysis
from ml.models import run_ml_pipeline
from explainability.shap_analyzer import run_shap_analysis
from visualization.visualizer import run_eda

logger = setup_logger(__name__)


class GeneExpressionPipeline:
    """Complete gene expression classification pipeline."""

    def __init__(self, config_path: str | Path | None = None):
        self.config = load_config(config_path)
        ensure_dirs(self.config)
        self.data = {}
        self.results = {}

    def run_all(self) -> dict[str, Any]:
        """Run the complete pipeline."""
        logger.info("=" * 70)
        logger.info("GENE EXPRESSION CLASSIFICATION PIPELINE")
        logger.info("=" * 70)

        # Step 1: Data Acquisition
        self._step_acquire_data()

        # Step 2: Exploratory Data Analysis
        self._step_eda()

        # Step 3: Preprocessing
        self._step_preprocessing()

        # Step 4: Feature Selection
        self._step_feature_selection()

        # Step 5: Biomarker Analysis
        self._step_biomarker_analysis()

        # Step 6: Machine Learning
        self._step_ml()

        # Step 7: SHAP Explainability
        self._step_shap()

        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        return self.results

    def _step_acquire_data(self) -> None:
        """Acquire the dataset."""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 1: DATA ACQUISITION")
        logger.info("=" * 50)

        df = acquire_dataset(self.config)
        
        # Ensure class column exists
        label_col = self.config["dataset"]["label_column"]
        if label_col not in df.columns:
            for col in df.columns:
                if 'class' in col.lower() or 'target' in col.lower():
                    label_col = col
                    break
        
        self.raw_data = df
        self.data["raw"] = df

        logger.info("Dataset loaded: %d samples x %d features", df.shape[0], df.shape[1])
        if label_col in df.columns:
            logger.info("Classes: %s", df[label_col].unique().tolist())

    def _step_eda(self) -> None:
        """Perform exploratory data analysis."""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 2: EXPLORATORY DATA ANALYSIS")
        logger.info("=" * 50)

        label_col = self.config["dataset"]["label_column"]
        if label_col in self.raw_data.columns:
            X = self.raw_data.drop(columns=[label_col])
            y = self.raw_data[label_col]
            run_eda(X, y, self.config)
        else:
            logger.warning("Label column not found. Skipping EDA.")

    def _step_preprocessing(self) -> None:
        """Run preprocessing pipeline."""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 3: PREPROCESSING")
        logger.info("=" * 50)

        X_train, X_test, y_train, y_test, scaler, feature_names = run_preprocessing(
            self.raw_data, self.config
        )

        self.data["X_train"] = X_train
        self.data["X_test"] = X_test
        self.data["y_train"] = y_train
        self.data["y_test"] = y_test
        self.data["scaler"] = scaler
        self.data["feature_names"] = feature_names

        logger.info("Preprocessing complete.")
        logger.info("  Train: %s", X_train.shape)
        logger.info("  Test: %s", X_test.shape)

    def _step_feature_selection(self) -> None:
        """Run feature selection."""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 4: FEATURE SELECTION")
        logger.info("=" * 50)

        try:
            features, scores = run_feature_selection(
                self.data["X_train"],
                self.data["y_train"],
                self.config
            )

            self.data["selected_features"] = features
            self.data["feature_scores"] = scores

            # Subset data to selected features
            self.data["X_train_selected"] = self.data["X_train"][features]
            self.data["X_test_selected"] = self.data["X_test"][features]

            logger.info("Selected %d features for modeling", len(features))
        except Exception as e:
            logger.warning("Feature selection failed: %s. Using all features.", str(e))
            self.data["X_train_selected"] = self.data["X_train"]
            self.data["X_test_selected"] = self.data["X_test"]

    def _step_biomarker_analysis(self) -> None:
        """Run biomarker discovery analysis."""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 5: BIOMARKER ANALYSIS")
        logger.info("=" * 50)

        try:
            analyzer = run_biomarker_analysis(
                self.data["X_train_selected"],
                self.data["y_train"],
                self.data["X_test_selected"],
                self.data["y_test"],
                self.config
            )
            self.results["biomarker_analyzer"] = analyzer
        except Exception as e:
            logger.warning("Biomarker analysis failed: %s", str(e))

    def _step_ml(self) -> None:
        """Run machine learning pipeline."""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 6: MACHINE LEARNING")
        logger.info("=" * 50)

        try:
            trainer = run_ml_pipeline(
                self.data["X_train_selected"],
                self.data["y_train"],
                self.data["X_test_selected"],
                self.data["y_test"],
                self.config
            )

            self.results["trainer"] = trainer
            self.results["best_model"] = trainer.best_model
            self.results["best_model_name"] = trainer.best_model_name
            self.data["model"] = trainer.best_model

            # Save best model
            model_dir = Path(self.config["paths"]["models"])
            model_dir.mkdir(parents=True, exist_ok=True)
            with open(model_dir / "best_model.pkl", "wb") as f:
                pickle.dump(trainer.best_model, f)

            comparison_df = trainer.get_model_comparison_df()
            logger.info("\nModel Comparison:")
            logger.info(comparison_df.to_string(index=False))
        except Exception as e:
            logger.error("ML pipeline failed: %s", str(e))

    def _step_shap(self) -> None:
        """Run SHAP explainability analysis."""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 7: SHAP EXPLAINABILITY")
        logger.info("=" * 50)

        if self.data.get("model") is None:
            logger.warning("No model available for SHAP analysis. Skipping.")
            return

        try:
            analyzer = run_shap_analysis(
                self.data["model"],
                self.data["X_train_selected"],
                self.data["X_test_selected"],
                self.config
            )
            self.results["shap_analyzer"] = analyzer
        except Exception as e:
            logger.warning("SHAP analysis failed: %s", str(e))


def main():
    """Main entry point for the pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Gene Expression-Based Disease Classification Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )
    args = parser.parse_args()

    # Run pipeline
    pipeline = GeneExpressionPipeline(args.config)

    try:
        results = pipeline.run_all()
        logger.info("\n✅ Pipeline completed successfully!")
        logger.info(f"Best model: {results.get('best_model_name', 'N/A')}")

        if "trainer" in results:
            comparison = results["trainer"].get_model_comparison_df()
            logger.info("\nModel comparison:")
            logger.info(comparison.to_string(index=False))

    except Exception as e:
        logger.error("Pipeline failed: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()