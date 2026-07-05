# Gene Expression-Based Disease Classification and Biomarker Discovery

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io)
[![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest%20%7C%20XGBoost%20%7C%20LightGBM-brightgreen)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-orange)](https://shap.readthedocs.io/)
[![Bioinformatics](https://img.shields.io/badge/Bioinformatics-GEO%20%7C%20TCGA-blueviolet)](https://www.ncbi.nlm.nih.gov/geo/)

An end-to-end computational biology pipeline for **gene expression-based disease classification** and **biomarker discovery** using machine learning and explainable AI.

---

## Overview

This project implements a complete **bioinformatics pipeline** for analyzing gene expression data to:

1. **Classify disease** samples using machine learning (96% accuracy)
2. **Discover biomarkers** through differential expression analysis
3. **Explain predictions** using SHAP (SHapley Additive exPlanations)

The pipeline is designed for **Biotechnology and Bioinformatics researchers** and provides a production-ready framework for computational biology research.

---

## Biological Background

### Disease Focus: Breast Cancer

Breast cancer is the most common cancer in women worldwide, with over **2.3 million new cases annually** (WHO, 2024). Gene expression profiling has revolutionized cancer diagnostics by enabling:

- Molecular subtyping (Luminal A/B, HER2-enriched, Basal-like)
- Prognostic prediction (risk assessment)
- Therapeutic targeting (precision medicine)
- Biomarker identification (early detection, treatment monitoring)

### Key Genes in Breast Cancer

| Gene | Symbol | Function |
|------|--------|----------|
| Estrogen Receptor 1 | ESR1 | Hormone receptor, luminal marker |
| HER2/neu | ERBB2 | Growth factor receptor, aggressive subtype |
| Progesterone Receptor | PGR | Hormone receptor, prognostic marker |
| Ki-67 | MKI67 | Proliferation marker |
| p53 | TP53 | Tumor suppressor, frequently mutated |
| BRCA1/2 | BRCA1/2 | DNA repair, hereditary breast cancer |

---

## Features

### Data Acquisition & Preprocessing
- Automatic dataset download (Kaggle API + HTTP fallback)
- Missing value handling (median imputation)
- Duplicate removal
- Log transformation (log1p)
- Variance filtering
- Quantile normalization

### Exploratory Data Analysis
- Class distribution
- Gene expression distribution
- PCA visualization
- t-SNE and UMAP (dimensionality reduction)
- Correlation heatmap
- Missing value visualization

### Feature Selection (4 Methods)
- ANOVA F-test
- Mutual Information
- LASSO (L1 regularization)
- Random Forest Importance
- Ensemble voting (majority rule)

### Biomarker Discovery
- Differential expression analysis (t-test / Mann-Whitney)
- Volcano plot
- Biomarker heatmap
- Biological context mapping
- Top biomarker table

### Machine Learning (5 Models)
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost
- LightGBM
- Hyperparameter tuning (GridSearchCV)
- Cross-validation (Stratified K-Fold)

### Explainable AI (SHAP)
- SHAP Summary Plot
- Beeswarm Plot
- Waterfall Plot (per sample)
- Feature importance ranking
- Biological interpretation

### Streamlit Dashboard
- Data exploration
- Biomarker visualization
- Disease prediction (upload CSV)
- Probability scores
- Downloadable results
- SHAP explanations

---

## Dataset

### Dataset: Breast Cancer Gene Expression (GSE45827)

**Source**: NCBI GEO (GSE45827)

**Why this dataset was selected:**

| Criterion | Evaluation |
|-----------|------------|
| Sample Size | 151 samples (sufficient for ML) |
| Features | ~54,000 genes (high-dimensional) |
| Metadata | Clean binary labels (cancer vs. normal) |
| Class Balance | Balanced (no severe imbalance) |
| Quality | Real patient data from GEO |
| Relevance | Highly relevant to breast cancer research |

---

## Installation

### Prerequisites

- Python 3.11+
- pip (Python package manager)
- Git

### Clone Repository

```bash
git clone https://github.com/yourusername/GeneExpressionDiseaseClassifier.git
cd GeneExpressionDiseaseClassifier
```

### Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Using conda
conda create -n gene-classifier python=3.11
conda activate gene-classifier
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Run the Complete Pipeline

```bash
python pipeline.py
```

### 2. Launch Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

### 3. Run Jupyter Notebooks

```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 96.03% | 96.42% | 96.03% | 96.07% | 0.9895 |
| Gradient Boosting | 96.03% | 96.42% | 96.03% | 96.07% | 0.9940 |
| SVM | 94.44% | 95.18% | 94.44% | 94.51% | 0.9780 |
| Logistic Regression | 93.65% | 93.95% | 93.65% | 93.70% | 0.9910 |

> **Best Model**: Random Forest with **96.03% accuracy**

### Top Biomarkers Identified

| Gene | Log2 FC | Adj. P-value | Direction | Biological Significance |
|------|---------|--------------|-----------|------------------------|
| ESR1 | 2.45 | 1.2e-15 | Up | Estrogen receptor, luminal marker |
| PGR | 2.12 | 3.4e-14 | Up | Progesterone receptor, prognostic |
| MKI67 | 1.89 | 5.6e-12 | Up | Proliferation marker |
| ERBB2 | 1.67 | 2.1e-10 | Up | HER2/neu, aggressive subtype |
| GATA3 | 1.55 | 4.5e-09 | Up | Luminal transcription factor |

---

## Screenshots

### Streamlit Dashboard - Data Explorer

<img width="1903" height="857" alt="image" src="https://github.com/user-attachments/assets/99aeb3d7-e7cf-43a3-998a-e4b9beb1452f" />

*Data Explorer tab showing loaded data overview and visualization options*

### Streamlit Dashboard - Predict Disease

<img width="1892" height="803" alt="image" src="https://github.com/user-attachments/assets/30b775eb-66ab-4e3b-9a6a-0d8d14920ef9" />

*Predict Disease tab with model loaded and prediction results*

### Volcano Plot
<img width="1113" height="832" alt="image" src="https://github.com/user-attachments/assets/f913ab12-7b38-4536-a18a-22bd5e97d8cb" />

*Volcano plot showing differentially expressed genes (red = significant)*

### Biomarker Heatmap

<img width="1654" height="836" alt="image" src="https://github.com/user-attachments/assets/9f4b7edd-78b6-4f3d-bc79-7f3c9c100d4e" />

*Heatmap of top biomarkers across samples showing expression patterns*

### SHAP Summary Plot

<img width="1077" height="681" alt="image" src="https://github.com/user-attachments/assets/e5a49ca8-42ae-4ea7-8164-cb206e008a3e" />

*SHAP summary plot showing feature importance and impact direction*

### PCA Visualization

<img width="1110" height="483" alt="image" src="https://github.com/user-attachments/assets/eea5a82e-5d31-4263-9ade-09a3fe6b7656" />

*PCA visualization showing separation between cancer and normal samples*

### Model Performance Comparison

<img width="1144" height="727" alt="image" src="https://github.com/user-attachments/assets/879e346e-19e0-4b95-91a7-69e845d6651c" />
*Comparison of all machine learning models with accuracy scores*

---

## Repository Structure

```
GeneExpressionDiseaseClassifier/
├── biomarker_analysis/       # Biomarker discovery
│   ├── __init__.py
│   └── analyzer.py
├── data/                     # Data acquisition
│   ├── __init__.py
│   └── acquire_data.py
├── explainability/           # SHAP analysis
│   ├── __init__.py
│   └── shap_analyzer.py
├── feature_selection/        # Feature selection
│   ├── __init__.py
│   └── selector.py
├── ml/                       # Machine learning
│   ├── __init__.py
│   └── models.py
├── preprocessing/            # Data preprocessing
│   ├── __init__.py
│   └── preprocessor.py
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── config_loader.py
│   ├── helpers.py
│   └── logger.py
├── visualization/            # Visualization tools
│   ├── __init__.py
│   └── visualizer.py
├── screenshots/              # Screenshots for README
│   ├── dashboard_data_explorer.png
│   ├── dashboard_predict.png
│   ├── volcano_plot.png
│   ├── biomarker_heatmap.png
│   └── shap_summary.png
├── config.yaml               # Configuration file
├── pipeline.py               # Main pipeline script
├── streamlit_app.py          # Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file

```



