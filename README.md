# Breast Cancer Classification - ML Assignment 2

## a. Problem Statement

The goal of this project is to build and compare multiple machine learning classification models for breast cancer diagnosis prediction. Using the Wisconsin Breast Cancer dataset, we implement 6 different classification algorithms and evaluate their performance using comprehensive evaluation metrics. The models are then deployed in an interactive Streamlit web application for real-world demonstration and prediction.

**Objective:**
- Implement 6 classification models on the breast cancer dataset
- Calculate comprehensive evaluation metrics for each model
- Deploy models in an interactive Streamlit web application
- Compare model performance to identify the most effective approach for medical diagnosis

---

## b. Dataset Description

**Dataset Overview:**
- **Name:** Wisconsin Breast Cancer Dataset
- **Source:** UCI Machine Learning Repository / Kaggle
- **Total Samples:** 569 (meets minimum requirement of 500)
- **Total Features:** 30 (meets minimum requirement of 12)
- **Target Variable:** Diagnosis (Binary Classification)
- **Classes:** 
  - Benign (B): 357 samples (62.7%)
  - Malignant (M): 212 samples (37.3%)

**Dataset Characteristics:**
- No missing values - dataset is clean and complete
- 30 computed features from digitized images of fine needle aspirates (FNA) of breast masses
- Features scaled using StandardScaler for normalization
- Train-Test Split: 80% training (455 samples), 20% testing (114 samples)
- Stratified split maintains class distribution in both training and testing sets

---

## c. Models Used: Performance Metrics and Observations

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Random Forest (Ensemble) | 0.9737 | 0.9929 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| XGBoost (Ensemble) | 0.9737 | 0.9940 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

**Evaluation Metrics:**
- **Accuracy:** Overall correctness of predictions
- **AUC:** Model's ability to distinguish between classes
- **Precision:** Ratio of correct positive predictions (minimizes false diagnoses)
- **Recall:** Ratio of actual positives correctly identified (minimizes missed cases)
- **F1-Score:** Harmonic mean balancing precision and recall
- **MCC:** Balanced metric for binary classification

---

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Excellent baseline model with exceptional AUC score of 0.9960 (highest among all models). Achieves 96.49% accuracy with high precision (0.9750) and strong recall (0.9286). The model effectively captures linear relationships and is highly interpretable, making it ideal for medical professionals. Computationally efficient with minimal training time. The high AUC demonstrates superior class discrimination despite slightly lower accuracy than ensemble methods. |
| **Decision Tree** | Moderate performance with 92.98% accuracy and lowest AUC (0.9246). Shows tendency to overfit on training data. While achieving balanced precision and recall (0.9048 each), the lower F1 score compared to other methods indicates suboptimal performance. Although highly interpretable for decision path visualization, the model is less reliable for medical diagnosis compared to ensemble methods. Demonstrates limitations of single-level decisions on complex medical data. |
| **KNN** | Solid performance with 95.61% accuracy and excellent AUC (0.9823). Achieves high precision (0.9744) with minimal false positives. Performs well with properly scaled features, validating that similar feature vectors have similar diagnoses. Local pattern recognition captures non-linear relationships effectively. Computationally efficient for inference but requires full training data in memory. The 0.9048 recall indicates effective identification of malignant cases. |
| **Naive Bayes** | Achieves 92.11% accuracy with surprisingly strong AUC (0.9891) - second highest among all models. The probabilistic approach works effectively despite theoretical feature independence assumptions not strictly holding. Maintains good precision (0.9231) but lower recall (0.8571), suggesting conservative positive predictions. Computationally efficient and interpretable, suitable for quick baseline classifications. MCC of 0.8292 indicates moderate balanced performance across both classes. |
| **Random Forest (Ensemble)** | Exceptional performance with 97.37% accuracy and 99.29% AUC. Achieves perfect precision (1.0000) - zero false positives, critical for medical diagnosis to avoid unnecessary procedures. Strong recall (0.9286) ensures minimal missed malignant cases. The ensemble approach aggregates multiple decision trees, reducing overfitting and capturing complex feature interactions. F1 score (0.9630) and MCC (0.9442) both indicate excellent balanced performance. Production-ready for reliable diagnosis. |
| **XGBoost (Ensemble)** | Matches Random Forest with 97.37% accuracy but achieves highest AUC among ensemble methods (0.9940). Perfect precision (1.0000) with excellent recall (0.9286) provides maximum diagnostic confidence. Gradient boosting builds trees sequentially, learning from previous errors for superior performance. F1 score (0.9630) and MCC (0.9442) indicate consistent high performance matching Random Forest. Most robust and production-ready model with superior generalization capabilities for clinical deployment. |



---

