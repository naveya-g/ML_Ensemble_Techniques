# 📡📉 ML Ensemble Techniques - Telecom Customer Churn Prediction

This project tackles high customer churn rates faced by a telecom company by building and comparing multiple machine learning models to predict which customers are likely to leave - enabling the company to take proactive retention action before it's too late.

---

### Project Objective

To identify customers who are at risk of leaving the telecom service, so the company can reach out to them with focused retention offers before they actually churn - turning a reactive problem into a proactive strategy.

---

### Dataset Summary

- Customer data from a telecom company split across 2 CSV files, merged on `customerID`
- **7,043 customer records** after merging
- Key features across 4 categories:
  - **Churn status** - whether the customer left within the last month
  - **Services subscribed** - Phone, MultipleLines, Internet, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
  - **Account information** - Contract type, PaymentMethod, PaperlessBilling, MonthlyCharges, TotalCharges, tenure
  - **Demographics** - Gender, SeniorCitizen, Partner, Dependents
- Target variable: **Churn** (Binary: Yes / No)
- Key challenges: 11 empty strings in `TotalCharges` column, `TotalCharges` stored as object type instead of float, class imbalance (majority non-churn)

---

### Analysis Workflow

- Merged 2 CSV files on `customerID`; replaced 11 empty strings in `TotalCharges` with 0 and converted it to float; dropped `customerID` as a non-predictive column
- Built a **pie chart function** to visualize all categorical features - found month-to-month contracts and fiber optic internet service strongly associated with higher churn
- Applied **label encoding** for ordinal categorical features and **one-hot encoding** for `PaymentMethod`; standardized all features using `StandardScaler`
- Performed 80:20 train-test split and trained 4 models (Decision Tree, Random Forest, AdaBoost, GradientBoost), each with and without **GridSearchCV hyperparameter tuning**

---

### Model Performance (Test Data — After Tuning)
 
| Model | Accuracy | Key Observation |
|---|---|---|
| Decision Tree | ~80.27% | Highest recall (0.649) — catches more churners but more false positives |
| Random Forest | ~81.3% | Highest precision (0.70) — fewest false alarms |
| **AdaBoost** ✅ | **81.41%** | **Best overall — strongest accuracy, good balance, no overfitting** |
| GradientBoost | ~81.12% | Stable performance, minimal change before/after tuning |
 
> **AdaBoost (tuned)** was selected as the final model — it achieved the highest test accuracy of **81.41%**, maintained consistency between train and test performance (no overfitting), and showed the best balance across precision, recall, and F1-score for churn detection.

---

### Tools Used

- Python, NumPy, Pandas, Matplotlib, Seaborn
- ML Models: `DecisionTreeClassifier`, `RandomForestClassifier`, `AdaBoostClassifier`, `GradientBoostingClassifier` (Scikit-learn)
- Hyperparameter Tuning: `GridSearchCV` with 5-fold cross-validation
- Preprocessing: `StandardScaler`, Label Encoding, One-Hot Encoding
- Evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- Environment: Google Colab

---

### 🔍 Use Case

The telecom company can use this model to score existing customers every month and flag those with a high probability of churning. These customers can then be targeted with personalized retention offers - such as contract upgrades, discounts, or bundled services - before they decide to leave, directly improving customer lifetime value and reducing churn-related revenue loss.
