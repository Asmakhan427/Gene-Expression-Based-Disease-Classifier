"""
analyzer.py - Biomarker discovery and analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless pipeline runs
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

from utils.logger import setup_logger

logger = setup_logger(__name__)


class BiomarkerAnalyzer:
    """Identify and analyze potential disease biomarkers."""

    def __init__(self, config):
        self.config = config
        self.biomarkers = None
        self.p_values = None          # aligned with self.feature_names (ALL features)
        self.fold_changes = None      # aligned with self.feature_names (ALL features)
        self.feature_names = None

    def find_biomarkers(self, X_train, y_train, X_test, y_test):
        """Identify biomarkers using statistical tests."""
        logger.info("Analyzing potential biomarkers...")

        self.feature_names = list(X_train.columns)
        classes = np.unique(y_train)

        if len(classes) != 2:
            logger.warning("Multi-class classification not yet supported for biomarker analysis")
            self.biomarkers = []
            return []

        class_0_mask = y_train == classes[0]
        class_1_mask = y_train == classes[1]

        X_class_0 = X_train[class_0_mask]
        X_class_1 = X_train[class_1_mask]

        p_values = []
        fold_changes = []

        for col in self.feature_names:
            try:
                _, p_val = ttest_ind(X_class_0[col], X_class_1[col])
            except Exception:
                p_val = 1.0
            p_values.append(p_val)

            mean_0 = X_class_0[col].mean()
            mean_1 = X_class_1[col].mean()
            fc = mean_1 / mean_0 if mean_0 > 0 else 0
            fold_changes.append(fc)

        self.p_values = np.array(p_values)
        self.fold_changes = np.array(fold_changes)

        threshold = self.config.get("biomarker", {}).get("p_threshold", 0.05)
        fc_threshold = self.config.get("biomarker", {}).get("fc_threshold", 1.0)

        significant = (self.p_values < threshold) & (
            np.abs(np.log2(self.fold_changes + 1e-6)) > fc_threshold
        )
        self.biomarkers = [f for f, sig in zip(self.feature_names, significant) if sig]

        logger.info(f"Found {len(self.biomarkers)} significant biomarkers")
        return self.biomarkers

    def get_biomarker_info(self):
        """Get detailed biomarker information, correctly aligned to each gene."""
        if not self.biomarkers:
            return None

        # Map each feature name to its position in the FULL feature list once
        name_to_idx = {name: i for i, name in enumerate(self.feature_names)}

        biomarker_data = []
        for biomarker in self.biomarkers:
            idx = name_to_idx[biomarker]   # correct lookup, not position-in-subset
            biomarker_data.append({
                "Biomarker": biomarker,
                "P-Value": self.p_values[idx],
                "Fold-Change": self.fold_changes[idx],
                "Log2FC": np.log2(self.fold_changes[idx] + 1e-6),
            })

        return pd.DataFrame(biomarker_data).sort_values("P-Value").reset_index(drop=True)

    def save_results(self, X_train, y_train, reports_dir):
        """Save all biomarker analysis outputs to reports/biomarkers/."""
        biomarker_dir = Path(reports_dir) / "biomarkers"
        biomarker_dir.mkdir(parents=True, exist_ok=True)

        info = self.get_biomarker_info()
        if info is None or info.empty:
            logger.warning("No significant biomarkers found; skipping file output.")
            return biomarker_dir

        # 1. top_biomarkers.csv
        info.to_csv(biomarker_dir / "top_biomarkers.csv", index=False)

        # 2. volcano_plot.png
        self._save_volcano_plot(biomarker_dir)

        # 3. biomarker_heatmap.png
        self._save_heatmap(X_train, y_train, info, biomarker_dir)

        # 4. biological_context.csv (placeholder scaffold — fill in with real
        #    annotation lookups, e.g. gene ontology / pathway DB, if available)
        context = info[["Biomarker", "Log2FC", "P-Value"]].copy()
        context.columns = ["Gene", "Log2FC", "P-Value"]
        context["Direction"] = ["Up" if fc > 0 else "Down" for fc in context["Log2FC"]]
        context["Description"] = "Not annotated"
        context["Pathway"] = "Unknown"
        context.to_csv(biomarker_dir / "biological_context.csv", index=False)

        # 5. biomarker_summary.txt
        with open(biomarker_dir / "biomarker_summary.txt", "w") as f:
            f.write("=" * 70 + "\n")
            f.write("BIOMARKER DISCOVERY SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total features screened: {len(self.feature_names)}\n")
            f.write(f"Significant biomarkers found: {len(self.biomarkers)}\n")
            f.write(f"P-value threshold: {self.config.get('biomarker', {}).get('p_threshold', 0.05)}\n")
            f.write(f"Fold-change threshold (log2): {self.config.get('biomarker', {}).get('fc_threshold', 1.0)}\n\n")
            f.write("Top 10 biomarkers by p-value:\n")
            f.write("-" * 70 + "\n")
            f.write(info.head(10).to_string(index=False))

        logger.info(f"✅ Saved biomarker results to: {biomarker_dir}")
        return biomarker_dir

    def _save_volcano_plot(self, biomarker_dir):
        """Generate and save volcano plot."""
        log2fc = np.log2(self.fold_changes + 1e-6)
        neg_log10_p = -np.log10(self.p_values + 1e-300)

        threshold = self.config.get("biomarker", {}).get("p_threshold", 0.05)
        fc_threshold = self.config.get("biomarker", {}).get("fc_threshold", 1.0)

        significant = (self.p_values < threshold) & (np.abs(log2fc) > fc_threshold)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(log2fc[~significant], neg_log10_p[~significant],
                   c="grey", alpha=0.5, s=20, label="Not significant")
        ax.scatter(log2fc[significant], neg_log10_p[significant],
                   c="red", alpha=0.7, s=30, label="Significant")
        ax.axhline(-np.log10(threshold), color="blue", linestyle="--", linewidth=1, label=f"p={threshold}")
        ax.axvline(fc_threshold, color="blue", linestyle="--", linewidth=1, alpha=0.5)
        ax.axvline(-fc_threshold, color="blue", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("log2(Fold Change)")
        ax.set_ylabel("-log10(p-value)")
        ax.set_title("Volcano Plot")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(biomarker_dir / "volcano_plot.png", dpi=150)
        plt.close(fig)
        logger.info("✅ Saved volcano_plot.png")

    def _save_heatmap(self, X_train, y_train, info, biomarker_dir, top_n=20):
        """Generate and save biomarker heatmap."""
        top_genes = info["Biomarker"].head(top_n).tolist()
        if not top_genes:
            return

        # Prepare data
        data = X_train[top_genes].copy()
        data["__class__"] = y_train.values
        data = data.sort_values("__class__")
        class_labels = data.pop("__class__")

        # Plot
        fig, ax = plt.subplots(figsize=(12, max(6, len(top_genes) * 0.3)))
        sns.heatmap(data.T, cmap="RdBu_r", center=0, ax=ax,
                   cbar_kws={"label": "Expression level (z-score)"})
        ax.set_xlabel("Samples")
        ax.set_ylabel("Genes")
        ax.set_title(f"Biomarker Expression Heatmap (Top {len(top_genes)} Genes)")
        fig.tight_layout()
        fig.savefig(biomarker_dir / "biomarker_heatmap.png", dpi=150)
        plt.close(fig)
        logger.info("✅ Saved biomarker_heatmap.png")


def run_biomarker_analysis(X_train, y_train, X_test, y_test, config):
    """
    Run biomarker discovery analysis and persist results to disk.

    Returns
    -------
    analyzer : BiomarkerAnalyzer
    """
    analyzer = BiomarkerAnalyzer(config)
    analyzer.find_biomarkers(X_train, y_train, X_test, y_test)
    analyzer.save_results(X_train, y_train, config["paths"]["reports"])
    return analyzer