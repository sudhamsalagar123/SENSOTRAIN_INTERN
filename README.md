# SENSOTRAIN_INTERN
Developed a machine learning model to predict customer churn using telecom data. Applied EDA, data preprocessing, feature engineering, and classification techniques to help the business identify at-risk customers and improve retention
# Customer Churn Prediction 📊

A machine learning project to predict customer churn using Random Forest classification. This project analyzes customer data to identify patterns that indicate whether a customer is likely to leave the service.

## 🎯 Project Overview

**Objective:** Predict whether a customer will churn (leave the service) based on features like contract type, payment method, tenure, and other customer characteristics.

**Dataset:** Customer data with features including demographics, account information, and service usage patterns.

## 🚀 Features

- **Data Exploration**: Comprehensive analysis of customer data
- **Data Cleaning**: Handles missing values and data type conversions
- **Feature Engineering**: One-hot encoding for categorical variables
- **Machine Learning**: Random Forest classifier implementation
- **Model Evaluation**: Performance metrics and confusion matrix
- **Feature Importance**: Visualization of key factors affecting churn
- **Model Persistence**: Save and load trained models

## 📋 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Ensure you have the dataset file `churn.csv` in the project directory.

## 📊 Dataset Structure

The dataset should contain the following types of columns:
- **Customer ID**: Unique identifier for each customer
- **Demographics**: Age, gender, senior citizen status
- **Account Information**: Contract type, payment method, tenure
- **Services**: Phone service, internet service, streaming services
- **Charges**: Monthly charges, total charges
- **Target**: Churn (Yes/No)

## 🔧 Usage

### Basic Usage

```python
# Run the complete pipeline
python churn_prediction.py
```

### Step-by-Step Execution

1. **Load and Explore Data**
```python
import pandas as pd
df = pd.read_csv("churn.csv")
df.head()
```

2. **Data Cleaning**
```python
# Handle missing values
df = df.dropna()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
```

3. **Feature Engineering**
```python
# Encode categorical variables
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, drop_first=True)
```

4. **Train Model**
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

5. **Evaluate Performance**
```python
from sklearn.metrics import classification_report, accuracy_score
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

## 📈 Model Performance

The Random Forest model provides:
- **Accuracy**: Typically 80-85% on test data
- **Precision/Recall**: Balanced performance for both churned and non-churned customers
- **Feature Importance**: Insights into key factors driving churn

## 📊 Key Insights

Common factors that influence customer churn:
- Contract type (month-to-month vs. long-term)
- Payment method
- Tenure (length of service)
- Monthly charges
- Total charges
- Service add-ons

## 🔍 Project Structure

```
customer-churn-prediction/
├── README.md
├── requirements.txt
├── churn_prediction.py
├── churn.csv
├── churn_model.pkl (generated after training)
└── visualizations/
    ├── feature_importance.png
    ├── confusion_matrix.png
    └── data_distribution.png
```

## 🎨 Visualizations

The project generates several visualizations:
- **Feature Importance Plot**: Shows which features most impact churn prediction
- **Confusion Matrix**: Visual representation of model performance
- **Data Distribution**: Histograms and count plots of key variables

## 🔧 Model Persistence

Save your trained model:
```python
import joblib
joblib.dump(model, 'churn_model.pkl')
```

Load the model for future predictions:
```python
loaded_model = joblib.load('churn_model.pkl')
predictions = loaded_model.predict(new_data)
```

## 🚀 Future Enhancements

- [ ] Implement additional algorithms (XGBoost, SVM, Neural Networks)
- [ ] Add hyperparameter tuning
- [ ] Create a web interface for predictions
- [ ] Add model interpretability with SHAP values
- [ ] Implement cross-validation
- [ ] Add automated model retraining pipeline

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

Your Name - [sudhamsalagar@gmail.com](mailto:your.email@example.com)

Project Link: [https://github.com/sudhamsalagar123/SENSOTRAIN_INTERN](https://github.com/sudhamsalagar123/customer-churn-prediction)

## 🙏 Acknowledgments

- Dataset source: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Inspiration from various churn prediction studies
- Thanks to the open-source community for the amazing tools

---

⭐ If you found this project helpful, please give it a star!
