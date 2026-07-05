# 🧬 Gene Expression-Based Disease Classification and Biomarker Discovery

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io)

An end-to-end computational biology pipeline for **gene expression-based disease classification** and **biomarker discovery** using machine learning and explainable AI.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Biological Background](#biological-background)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Steps](#pipeline-steps)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Visualizations](#visualizations)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## Overview

This project implements a complete **bioinformatics pipeline** for analyzing gene expression data to:

1. **Classify disease** samples using machine learning
2. **Discover biomarkers** through differential expression analysis
3. **Explain predictions** using SHAP (SHapley Additive exPlanations)

The pipeline is designed for **Biotechnology and Bioinformatics researchers** and provides a production-ready framework for computational biology research.

---

## Biological Background

### Disease Focus: **Breast Cancer**

Breast cancer is the most common cancer in women worldwide, with over 2.3 million new cases annually (WHO, 2024). Gene expression profiling has revolutionized cancer diagnostics by enabling:

- **Molecular subtyping** (Luminal A/B, HER2-enriched, Basal-like)
- **Prognostic prediction** (risk assessment)
- **Therapeutic targeting** (precision medicine)
- **Biomarker identification** (early detection, treatment monitoring)

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

## Dataset

### Dataset: Breast Cancer Gene Expression (GSE45827)

**Source**: Kaggle ([brunogrisci/breast-cancer-gene-expression-data](https://www.kaggle.com/datasets/brunogrisci/breast-cancer-gene-expression-data))

**Original Source**: NCBI GEO (GSE45827)

**Why this dataset was selected:**

| Criterion | Evaluation |
|-----------|------------|
| **Sample Size** | 151 samples (sufficient for ML) |
| **Features** | ~54,000 genes (high-dimensional) |
| **Metadata** | Clean binary labels (cancer vs. normal) |
| **Class Balance** | Balanced (no severe imbalance) |
| **Quality** | Real patient data from GEO |
| **Relevance** | Highly relevant to breast cancer research |

---

## Features

### Data Acquisition & Preprocessing
- ✅ Automatic dataset download (Kaggle API + HTTP fallback)
- ✅ Missing value handling (median imputation)
- ✅ Duplicate removal
- ✅ Log transformation (log1p)
- ✅ Variance filtering
- ✅ Quantile normalization

### Exploratory Data Analysis
- ✅ Class distribution
- ✅ Gene expression distribution
- ✅ PCA visualization
- ✅ t-SNE and UMAP (dimensionality reduction)
- ✅ Correlation heatmap
- ✅ Missing value visualization

### Feature Selection (4 Methods)
- ✅ ANOVA F-test
- ✅ Mutual Information
- ✅ LASSO (L1 regularization)
- ✅ Random Forest Importance
- ✅ Ensemble voting (majority rule)

### Biomarker Discovery
- ✅ Differential expression analysis (t-test / Mann-Whitney)
- ✅ Volcano plot
- ✅ Biomarker heatmap
- ✅ Biological context mapping
- ✅ Top biomarker table

### Machine Learning (5 Models)
- ✅ Logistic Regression
- ✅ Support Vector Machine (SVM)
- ✅ Random Forest
- ✅ XGBoost
- ✅ LightGBM
- ✅ Hyperparameter tuning (GridSearchCV)
- ✅ Cross-validation (Stratified K-Fold)

### Explainable AI (SHAP)
- ✅ SHAP Summary Plot
- ✅ Beeswarm Plot
- ✅ Waterfall Plot (per sample)
- ✅ Force Plot
- ✅ Feature importance ranking
- ✅ Biological interpretation

### Streamlit Dashboard
- ✅ Data exploration
- ✅ Biomarker visualization
- ✅ Disease prediction (upload CSV)
- ✅ Probability scores
- ✅ Downloadable results
- ✅ SHAP explanations

---

## Installation

### Prerequisites

- Python 3.11+
- pip (Python package manager)
- Git

### Clone Repository

```bash
git clone https://github.com/somiyakhan10/GeneExpressionDiseaseClassifier.git
cd GeneExpressionDiseaseClassifier