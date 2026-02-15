"""
Streamlit Web Application for Breast Cancer Classification
Interactive dashboard for evaluating test data using trained ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                            accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, matthews_corrcoef)
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load models function
@st.cache_resource
def load_all_models():
    """Load all trained models from disk"""
    models = {}
    model_dir = 'model'
    
    model_files = {
        'Logistic Regression': 'Logistic_Regression_model.pkl',
        'Decision Tree': 'Decision_Tree_model.pkl',
        'KNN': 'KNN_model.pkl',
        'Naive Bayes': 'Naive_Bayes_model.pkl',
        'Random Forest': 'Random_Forest_model.pkl',
        'XGBoost': 'XGBoost_model.pkl'
    }
    
    for model_name, filename in model_files.items():
        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                models[model_name] = pickle.load(f)
        else:
            st.warning(f"Model {model_name} not found at {filepath}")
    
    return models

def scale_features(data):
    """Scale features using StandardScaler"""
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def evaluate_model(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'AUC Score': roc_auc_score(y_true, y_pred_proba),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    return metrics

def create_sample_dataset():
    """Create a sample test dataset from test data (50 samples)"""
    try:
        df = pd.read_csv('test_data.csv')
        # Get 50 random samples from test data
        sample_df = df.sample(n=min(50, len(df)), random_state=42)
        return sample_df
    except:
        return None

def get_dataset_info():
    """Get information about the training dataset"""
    try:
        df = pd.read_csv(r'c:\Users\nagab\Downloads\archive (4)\breast-cancer.csv')
        total_samples = len(df)
        total_features = len(df.columns) - 2  # Exclude id and diagnosis
        
        if 'diagnosis' in df.columns:
            diagnosis_counts = df['diagnosis'].value_counts()
            info = {
                'total_samples': total_samples,
                'total_features': total_features,
                'diagnosis_counts': diagnosis_counts.to_dict()
            }
            return info
    except:
        return None

def evaluate_all_models(X_test, y_test, models_dict):
    """Evaluate all models on test data and return comparison results"""
    results = {}
    X_test_scaled = scale_features(X_test)
    
    for model_name, model in models_dict.items():
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_pred_proba)
        results[model_name] = metrics
    
    return pd.DataFrame(results).T

# Main app
def main():
    st.markdown("<h1 class='main-header'>🏥 Breast Cancer Classification System</h1>", 
                unsafe_allow_html=True)
    
    st.markdown("""
    Upload your test data (CSV) to evaluate it using trained machine learning models 
    for breast cancer diagnosis prediction.
    """)
    
    # Load models
    models = load_all_models()
    
    if not models:
        st.error("No trained models found! Please run train_models.ipynb first.")
        return
    
    # Dataset Information Section
    st.header("Dataset Information")
    dataset_info = get_dataset_info()
    
    if dataset_info:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", dataset_info['total_samples'])
        with col2:
            st.metric("Total Features", dataset_info['total_features'])
        with col3:
            st.metric("Classes", len(dataset_info['diagnosis_counts']))
        
        st.subheader("Class Distribution")
        col1, col2 = st.columns(2)
        with col1:
            for diagnosis, count in dataset_info['diagnosis_counts'].items():
                percentage = (count / dataset_info['total_samples']) * 100
                st.write(f"**{diagnosis}:** {count} samples ({percentage:.1f}%)")
        
        with col2:
            st.subheader("Sample Dataset Download")
            sample_data = create_sample_dataset()
            if sample_data is not None:
                csv = sample_data.to_csv(index=False)
                st.download_button(
                    label="Download Sample Test Data",
                    data=csv,
                    file_name="sample_test_data.csv",
                    mime="text/csv"
                )
                st.caption("(50 random samples from the test dataset)")
    
    st.divider()
    st.header("Model Comparison")
    st.info("Performance metrics across all 6 trained models on the test dataset")
    
    try:
        # Load test data for model comparison
        test_data = pd.read_csv('test_data.csv')
        X_comparison = test_data.drop(['diagnosis', 'id'], axis=1, errors='ignore')
        y_comparison_raw = test_data['diagnosis']
        
        # Convert diagnosis labels to numeric
        if y_comparison_raw.dtype == 'object':
            y_comparison = y_comparison_raw.map({'B': 0, 'M': 1, 'Benign': 0, 'Malignant': 1})
        else:
            y_comparison = y_comparison_raw
        
        with st.spinner("Evaluating all models..."):
            comparison_df = evaluate_all_models(X_comparison, y_comparison, models)
            
            # Sort by accuracy for better readability
            comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
            
            # Display comparison table
            st.dataframe(
                comparison_df.style.format("{:.4f}").background_gradient(cmap="RdYlGn", axis=0),
                use_container_width=True
            )
            
            # Metrics visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Accuracy Comparison")
                accuracy_data = comparison_df['Accuracy'].sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, 6))
                accuracy_data.plot(kind='barh', ax=ax, color='steelblue')
                ax.set_xlabel('Accuracy Score', fontsize=11, fontweight='bold')
                ax.set_ylabel('Model', fontsize=11, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.subheader("F1-Score Comparison")
                f1_data = comparison_df['F1 Score'].sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, 6))
                f1_data.plot(kind='barh', ax=ax, color='coral')
                ax.set_xlabel('F1 Score', fontsize=11, fontweight='bold')
                ax.set_ylabel('Model', fontsize=11, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
            
            # Download comparison results
            st.divider()
            comparison_csv = comparison_df.to_csv()
            st.download_button(
                label="Download Model Comparison as CSV",
                data=comparison_csv,
                file_name="model_comparison.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.warning(f"Could not load test data for comparison: {str(e)}")
    
    st.divider()
    st.header("Upload and Evaluate Test Data")
    st.info("Upload a CSV with 30 features and a 'diagnosis' column (Benign or Malignant)")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="test_data_upload")
    
    if uploaded_file is not None:
        try:
            # Read uploaded data
            uploaded_data = pd.read_csv(uploaded_file)
            
            st.subheader("Uploaded Data Preview")
            st.dataframe(uploaded_data.head(10), use_container_width=True)
            st.info(f"Data shape: {uploaded_data.shape[0]} rows × {uploaded_data.shape[1]} columns")
            
            # Check if diagnosis column exists
            if 'diagnosis' not in uploaded_data.columns:
                st.error("No 'diagnosis' column found. Your CSV must have a 'diagnosis' column.")
            else:
                # Extract features and target
                X_test = uploaded_data.drop(['diagnosis', 'id'], axis=1, errors='ignore')
                y_test_raw = uploaded_data['diagnosis']
                
                # Convert diagnosis labels to numeric (B=0, M=1) if they are strings
                if y_test_raw.dtype == 'object':
                    y_test = y_test_raw.map({'B': 0, 'M': 1, 'Benign': 0, 'Malignant': 1})
                    if y_test.isnull().any():
                        st.error("Invalid diagnosis values. Expected 'B'/'M' or 0/1")
                        st.stop()
                else:
                    y_test = y_test_raw
                
                st.divider()
                st.subheader("Model Selection & Evaluation")
                
                selected_model_name = st.selectbox(
                    "Select a model to evaluate:",
                    list(models.keys()),
                    key="eval_model_selection"
                )
                
                if st.button("Evaluate Model", type="primary"):
                    with st.spinner("Evaluating model..."):
                        # Scale features
                        X_test_scaled = scale_features(X_test)
                        
                        # Get model and predictions
                        model = models[selected_model_name]
                        y_pred = model.predict(X_test_scaled)
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                        
                        # Calculate metrics
                        metrics = evaluate_model(y_test, y_pred, y_pred_proba)
                        
                        # Display metrics
                        st.markdown("<div class='success-box'>", unsafe_allow_html=True)
                        st.success(f"Evaluation Complete for {selected_model_name}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Metrics columns
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                        with col2:
                            st.metric("Precision", f"{metrics['Precision']:.4f}")
                        with col3:
                            st.metric("Recall", f"{metrics['Recall']:.4f}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                        with col2:
                            st.metric("AUC Score", f"{metrics['AUC Score']:.4f}")
                        with col3:
                            st.metric("MCC", f"{metrics['MCC']:.4f}")
                        
                        st.divider()
                        
                        # Confusion Matrix
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Confusion Matrix")
                            cm = confusion_matrix(y_test, y_pred)
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                       xticklabels=['Benign', 'Malignant'],
                                       yticklabels=['Benign', 'Malignant'],
                                       cbar_kws={'label': 'Count'})
                            ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
                            ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        with col2:
                            st.subheader("Classification Report")
                            report_dict = classification_report(y_test, y_pred, 
                                                               output_dict=True)
                            report_df = pd.DataFrame(report_dict).T
                            st.dataframe(
                                report_df.style.format("{:.4f}"),
                                use_container_width=True
                            )
                        
                        st.divider()
                        
                        # Prediction details
                        st.subheader("Detailed Predictions")
                        
                        predictions_df = pd.DataFrame({
                            'Actual': y_test.values,
                            'Predicted': y_pred,
                            'Confidence': np.max(model.predict_proba(X_test_scaled), axis=1),
                            'Probability (Benign)': model.predict_proba(X_test_scaled)[:, 0],
                            'Probability (Malignant)': model.predict_proba(X_test_scaled)[:, 1],
                            'Correct': y_test.values == y_pred
                        })
                        
                        # Show class mapping
                        predictions_df['Actual'] = predictions_df['Actual'].map({0: 'Benign', 1: 'Malignant'})
                        predictions_df['Predicted'] = predictions_df['Predicted'].map({0: 'Benign', 1: 'Malignant'})
                        
                        st.dataframe(
                            predictions_df.style.format({
                                'Confidence': '{:.4f}',
                                'Probability (Benign)': '{:.4f}',
                                'Probability (Malignant)': '{:.4f}'
                            }),
                            use_container_width=True
                        )
                        
                        # Download results
                        st.divider()
                        st.subheader("Download Results")
                        
                        results_export = predictions_df.copy()
                        csv = results_export.to_csv(index=False)
                        st.download_button(
                            label="Download Evaluation Results as CSV",
                            data=csv,
                            file_name=f"evaluation_{selected_model_name.replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
                        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About This Application:**
    
    This Streamlit app evaluates trained machine learning models on your test data.
    
    **Models Available:** 6 (Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost)  
    **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, AUC, Matthews Correlation Coefficient (MCC)
    
    **Upload Format:**  
    Your CSV file should contain 30 feature columns and a 'diagnosis' column with Benign or Malignant values.
    """)

if __name__ == "__main__":
    main()
