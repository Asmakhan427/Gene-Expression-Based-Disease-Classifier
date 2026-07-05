"""
streamlit_app.py
=================
Interactive web application for the gene expression classifier.
"""

import sys
import pickle
import warnings
from pathlib import Path

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config_loader import load_config
from preprocessing.preprocessor import load_processed
from ml.models import ModelTrainer

# ============================================
# DARK BLUE THEME WITH WHITE TEXT
# ============================================
st.markdown("""
<style>
    /* Sidebar - Dark Blue Background */
    .css-1d391kg, .css-1lcbmhc, .st-emotion-cache-1d391kg {
        background-color: #0a1628 !important;
    }
    
    /* ALL SIDEBAR TEXT - WHITE */
    .st-emotion-cache-1d391kg,
    .st-emotion-cache-1d391kg *,
    .css-1d391kg,
    .css-1d391kg *,
    .css-1lcbmhc,
    .css-1lcbmhc * {
        color: #ffffff !important;
    }
    
    /* Sidebar paragraphs */
    .st-emotion-cache-1d391kg p,
    .css-1d391kg p,
    .css-1lcbmhc p {
        color: #ffffff !important;
    }
    
    /* Sidebar labels */
    .st-emotion-cache-1d391kg label,
    .css-1d391kg label,
    .css-1lcbmhc label {
        color: #ffffff !important;
    }
    
    /* Sidebar radio buttons */
    .st-emotion-cache-1d391kg .stRadio label,
    .css-1d391kg .stRadio label {
        color: #ffffff !important;
    }
    
    .st-emotion-cache-1d391kg .stRadio [role="radiogroup"] label[data-checked="true"],
    .css-1d391kg .stRadio [role="radiogroup"] label[data-checked="true"] {
        color: #4a8db7 !important;
        font-weight: bold !important;
    }
    
    /* Sidebar headers - Light Blue accent */
    .st-emotion-cache-1d391kg h1,
    .st-emotion-cache-1d391kg h2,
    .st-emotion-cache-1d391kg h3,
    .st-emotion-cache-1d391kg h4,
    .css-1d391kg h1,
    .css-1d391kg h2,
    .css-1d391kg h3,
    .css-1d391kg h4 {
        color: #4a8db7 !important;
    }
    
    /* Main content background */
    .stApp {
        background-color: #0a1628 !important;
    }
    
    /* Main content text */
    .stApp .stMarkdown p,
    .stApp .stMarkdown li,
    .stApp label,
    .stApp .stText {
        color: #e8f0fe !important;
    }
    
    /* Main content headers */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #4a8db7 !important;
    }
    
    /* Metric labels */
    .stMetric label {
        color: #4a8db7 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    
    /* Metric values */
    .stMetric .stMetricValue {
        color: #ffffff !important;
        font-size: 28px !important;
        font-weight: 600 !important;
    }
    
    /* Metric container */
    .stMetric {
        background-color: #13294b !important;
        border: 1px solid #1a3a5c !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #1a3a5c !important;
        color: #ffffff !important;
        border: 1px solid #4a8db7 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #2a5a7c !important;
        border-color: #6aa8d7 !important;
        box-shadow: 0 0 20px rgba(74, 141, 183, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Dataframes */
    .dataframe {
        background-color: #0a1628 !important;
        color: #e8f0fe !important;
    }
    
    .dataframe thead tr th {
        background-color: #1a3a5c !important;
        color: #ffffff !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: #13294b !important;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background-color: #13294b !important;
        border: 2px dashed #1a3a5c !important;
        border-radius: 12px !important;
    }
    
    .stFileUploader > div:hover {
        border-color: #4a8db7 !important;
    }
    
    /* Success/Warning/Info boxes */
    .stAlert {
        background-color: #13294b !important;
        border-left-color: #4a8db7 !important;
        border-radius: 8px !important;
    }
    
    .stAlert .stMarkdown p {
        color: #e8f0fe !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #13294b !important;
        color: #e8f0fe !important;
        border: 1px solid #1a3a5c !important;
        border-radius: 8px !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #13294b !important;
        color: #e8f0fe !important;
        border-radius: 8px 8px 0px 0px;
        padding: 0.5rem 1rem !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1a3a5c !important;
        border-bottom: 3px solid #4a8db7 !important;
        color: #4a8db7 !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a1628;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #1a3a5c;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #4a8db7;
    }
</style>
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="Gene Expression Disease Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


class StreamlitApp:
    """Streamlit application for gene expression classification."""

    def __init__(self):
        self.config = load_config()
        self.data = None
        self.model = None
        self.model_name = None

    def load_saved_model(self) -> bool:
        """Load the saved best model."""
        try:
            model_dir = Path(self.config["paths"]["models"])
            model_path = model_dir / "best_model.pkl"

            if not model_path.exists():
                model_files = list(model_dir.glob("*.pkl"))
                if model_files:
                    model_path = model_files[0]
                else:
                    st.warning("No saved models found. Please run the pipeline first.")
                    return False

            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.model_name = model_path.stem
            return True
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return False

    def load_preprocessed_data(self) -> bool:
        """Load preprocessed data."""
        try:
            self.data = load_processed()
            return True
        except FileNotFoundError:
            st.warning("No preprocessed data found. Please run the pipeline first.")
            return False
        except Exception as e:
            st.error(f"Failed to load preprocessed data: {str(e)}")
            return False

    def run(self):
        """Run the Streamlit application."""
        st.title("🧬 Gene Expression-Based Disease Classifier")
        st.markdown("""
        An end-to-end bioinformatics pipeline for disease classification
        and biomarker discovery using Machine Learning and Explainable AI.
        """)

        # Sidebar
        with st.sidebar:
            st.markdown("### Navigation")
            app_mode = st.radio(
                "Choose a mode:",
                ["Data Explorer", "Biomarker Discovery", "Predict Disease"],
            )

        # Main content
        if app_mode == "Data Explorer":
            self.data_explorer()
        elif app_mode == "Biomarker Discovery":
            self.biomarker_discovery()
        else:
            self.predict_disease()

    def data_explorer(self):
        """Data exploration mode."""
        st.header("Data Explorer")

        if not self.load_preprocessed_data():
            return

        st.subheader("Loaded Data Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", self.data["X_train"].shape[0])
        with col2:
            st.metric("Test Samples", self.data["X_test"].shape[0])
        with col3:
            st.metric("Features", len(self.data["feature_names"]))

        st.subheader("Upload Your Own Data")
        uploaded_file = st.file_uploader(
            "Upload gene expression data (CSV format)",
            type=["csv"],
            help="Upload a CSV file with samples as rows and genes as columns",
        )

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded data: {df.shape[0]} samples × {df.shape[1]} features")
            st.subheader("Data Preview")
            st.dataframe(df.head())

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Samples", df.shape[0])
            with col2:
                st.metric("Features", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())

            if df.shape[1] > 2:
                st.subheader("Data Visualization")
                gene = st.selectbox("Select a gene to visualize", df.columns)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(df[gene].dropna(), bins=30, alpha=0.7, color='#4a8db7', edgecolor='#6aa8d7')
                ax.set_xlabel("Expression level", color='#ffffff', fontsize=12)
                ax.set_ylabel("Frequency", color='#ffffff', fontsize=12)
                ax.set_title(f"Distribution of {gene}", color='#4a8db7', fontsize=14, fontweight='bold')
                ax.tick_params(colors='#ffffff', labelsize=10)
                for spine in ax.spines.values():
                    spine.set_color('#4a8db7')
                ax.set_facecolor('#0a1628')
                fig.patch.set_facecolor('#0a1628')
                ax.grid(True, alpha=0.2, color='#4a8db7')
                st.pyplot(fig)

    def biomarker_discovery(self):
        """Biomarker discovery mode."""
        st.header("Biomarker Discovery")

        if not self.load_preprocessed_data():
            return

        biomarker_dir = Path(self.config["paths"]["reports"]) / "biomarkers"
        if not biomarker_dir.exists():
            st.info("Biomarker results not found. Running biomarker analysis...")
            try:
                from biomarker_analysis.analyzer import run_biomarker_analysis
                data = self.data
                X_train = data.get('X_train')
                X_test = data.get('X_test')
                y_train = data.get('y_train')
                y_test = data.get('y_test')
                if X_train is not None:
                    run_biomarker_analysis(X_train, y_train, X_test, y_test, self.config)
                    st.success("Biomarker analysis completed!")
                else:
                    st.warning("Could not run biomarker analysis. Please run the pipeline first.")
                    return
            except Exception as e:
                st.warning(f"Could not run biomarker analysis: {str(e)}. Please run the pipeline first.")
                return

        top_file = biomarker_dir / "top_biomarkers.csv"
        if top_file.exists():
            top_biomarkers = pd.read_csv(top_file)
            st.subheader("Top Biomarkers")
            st.dataframe(top_biomarkers.head(20))

            col1, col2 = st.columns(2)
            with col1:
                volcano_plot = biomarker_dir / "volcano_plot.png"
                if volcano_plot.exists():
                    st.subheader("Volcano Plot")
                    st.image(str(volcano_plot))

            with col2:
                heatmap = biomarker_dir / "biomarker_heatmap.png"
                if heatmap.exists():
                    st.subheader("Biomarker Heatmap")
                    st.image(str(heatmap))

        context_file = biomarker_dir / "biological_context.csv"
        if context_file.exists():
            st.subheader("Biological Context")
            context = pd.read_csv(context_file)
            st.dataframe(context)

    def predict_disease(self):
        """Disease prediction mode."""
        st.header("Predict Disease")

        if not self.load_saved_model():
            return

        if not self.load_preprocessed_data():
            return

        st.success(f"✅ Loaded model: {self.model_name}")

        uploaded_file = st.file_uploader(
            "Upload gene expression data for prediction",
            type=["csv"],
            help="Upload a CSV file with gene expression values",
        )

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())

                expected_features = self.data["feature_names"]
                missing_features = set(expected_features) - set(df.columns)

                if missing_features:
                    st.warning(
                        f"Missing {len(missing_features)} features. "
                        "Using available features for prediction."
                    )

                if st.button("Predict Disease", type="primary"):
                    with st.spinner("Making predictions..."):
                        if all(f in df.columns for f in expected_features):
                            X_new = df[expected_features]
                        else:
                            available = [f for f in expected_features if f in df.columns]
                            X_new = df[available]
                            st.info(f"Using {len(available)} available features")

                        scaler = self.data["scaler"]
                        X_scaled = scaler.transform(X_new.values)

                        predictions = self.model.predict(X_scaled)
                        probabilities = self.model.predict_proba(X_scaled)

                        if hasattr(self.model, 'classes_'):
                            classes = self.model.classes_
                        else:
                            classes = ['normal', 'cancer']

                        st.subheader("Prediction Results")

                        results_df = pd.DataFrame({
                            "Sample": range(1, len(predictions) + 1),
                            "Prediction": predictions,
                            "Confidence": probabilities.max(axis=1),
                            "Probability (Normal)": probabilities[:, 0],
                            "Probability (Cancer)": probabilities[:, 1] if probabilities.shape[1] > 1 else None,
                        })

                        st.dataframe(results_df.style.background_gradient(subset=['Confidence']))

                        st.subheader("Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            normal_count = sum(predictions == 'normal')
                            st.metric("Normal", normal_count)
                        with col2:
                            cancer_count = sum(predictions == 'cancer')
                            st.metric("Cancer", cancer_count)
                        with col3:
                            avg_confidence = probabilities.max(axis=1).mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.2%}")

                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions (CSV)",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv",
                        )

            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")


def main():
    """Main entry point for Streamlit application."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()