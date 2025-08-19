# 🚢 Titanic Survival Prediction - EDA & Machine Learning

A comprehensive exploratory data analysis and machine learning project using the famous Titanic dataset to predict passenger survival rates.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Methodology](#methodology)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Future Improvements](#future-improvements)

## 🎯 Project Overview

This project analyzes the Titanic passenger dataset to understand the factors that influenced survival during the 1912 disaster. Through comprehensive exploratory data analysis (EDA) and machine learning modeling, we identify key patterns and build predictive models.

**Main Goals:**
- Perform thorough exploratory data analysis
- Identify factors that influenced passenger survival
- Build and compare multiple machine learning models
- Provide actionable insights about survival patterns

## 📊 Dataset

The Titanic dataset contains information about 891 passengers with the following features:

| Feature | Description | Type |
|---------|-------------|------|
| PassengerId | Unique passenger identifier | Numeric |
| Survived | Survival status (0 = No, 1 = Yes) | Binary |
| Pclass | Passenger class (1st, 2nd, 3rd) | Categorical |
| Name | Passenger name | Text |
| Sex | Gender (male/female) | Categorical |
| Age | Age in years | Numeric |
| SibSp | Number of siblings/spouses aboard | Numeric |
| Parch | Number of parents/children aboard | Numeric |
| Ticket | Ticket number | Text |
| Fare | Passenger fare | Numeric |
| Cabin | Cabin number | Text |
| Embarked | Port of embarkation (C/Q/S) | Categorical |

## 🛠 Installation

### Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the notebook file
3. Upload your Titanic dataset CSV file to `/content/`
4. Run all cells

### Local Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/titanic-eda-ml.git
cd titanic-eda-ml

# Install required packages
pip install pandas numpy seaborn matplotlib scikit-learn

# Run Jupyter Notebook
jupyter notebook
```

## 📁 Project Structure

```
titanic-eda-ml/
├── README.md
├── titanic_eda_analysis.ipynb
├── data/
│   ├── train.csv
│   └── Titanic-Dataset.csv
├── visualizations/
│   ├── univariate_plots.png
│   ├── bivariate_plots.png
│   └── correlation_heatmap.png
└── results/
    ├── model_performance.txt
    └── feature_importance.png
```

## 🔍 Key Findings

### Survival Statistics
- **Overall survival rate: 38%** (342 survived out of 891 passengers)
- **Gender impact:** Females had 74% survival rate vs. males at 19%
- **Class impact:** 1st class (63%) > 2nd class (47%) > 3rd class (24%)

### Critical Factors for Survival

1. **Gender** - Most important factor
   - "Women and children first" policy was clearly implemented
   
2. **Passenger Class & Fare** 
   - Higher class passengers had better access to lifeboats
   - Strong correlation between fare paid and survival chances
   
3. **Age**
   - Children under 16 had higher survival rates
   - Elderly passengers had lower survival rates
   
4. **Family Size**
   - Passengers with 1-2 family members had optimal survival
   - Large families (5+) and solo travelers had lower survival
   
5. **Port of Embarkation**
   - Cherbourg (C): 55% survival rate
   - Queenstown (Q): 39% survival rate  
   - Southampton (S): 33% survival rate

## 🔬 Methodology

### 1. Data Preprocessing
- **Missing Values:** 
  - Age: Filled with median (robust to outliers)
  - Embarked: Dropped 2 rows with missing values
  - Cabin: Dropped entire column (77% missing)
- **Data Types:** Converted categorical variables for memory efficiency
- **Duplicates:** Verified no duplicate records exist

### 2. Exploratory Data Analysis

#### Univariate Analysis
- Distribution of each feature individually
- Identified data skewness and outliers
- Assessed class imbalances

#### Bivariate Analysis  
- Survival rates across different categories
- Box plots for continuous variables vs survival
- Cross-tabulation analysis

#### Multivariate Analysis
- Correlation matrix of numeric features
- Combined effect of multiple features on survival
- Feature interaction analysis

### 3. Machine Learning Pipeline
- **Models Tested:**
  - Logistic Regression
  - Decision Tree Classifier  
  - Random Forest Classifier
- **Evaluation Metrics:**
  - Accuracy Score
  - Confusion Matrix
  - Classification Report
  - Cross-Validation (5-fold)

## 📈 Results

### Model Performance
| Model | Accuracy | Cross-Val Mean |
|-------|----------|----------------|
| Random Forest | ~82-84% | ~81% |
| Logistic Regression | ~78-80% | ~79% |
| Decision Tree | ~75-77% | ~76% |

### Feature Importance (Random Forest)
1. **Sex** - Most predictive feature
2. **Fare** - Strong predictor of survival
3. **Age** - Moderate importance  
4. **Pclass** - Important class indicator
5. **Family Size** - Minor but useful predictor

## 💻 Technologies Used

- **Python 3.7+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization
- **Scikit-learn** - Machine learning algorithms
- **Google Colab** - Cloud-based Jupyter environment

## 🚀 How to Run

### Option 1: Google Colab (Easiest)
1. Download the notebook file
2. Open [Google Colab](https://colab.research.google.com/)
3. Upload the notebook
4. Upload your `train.csv` or `Titanic-Dataset.csv` to `/content/`
5. Run all cells sequentially

### Option 2: Local Jupyter
1. Install required packages: `pip install -r requirements.txt`
2. Place dataset in the project folder
3. Open Jupyter: `jupyter notebook`
4. Run the notebook

### Getting the Dataset
- **Kaggle API:** `!kaggle competitions download -c titanic`
- **Direct Download:** [Titanic Dataset on Kaggle](https://www.kaggle.com/c/titanic/data)
- **Manual Upload:** Upload CSV file directly to Colab

## 🔮 Future Improvements

### Feature Engineering
- [ ] Extract titles from passenger names (Mr., Mrs., Miss, etc.)
- [ ] Create age groups/bins for better categorical analysis
- [ ] Engineer deck information from cabin numbers
- [ ] Create fare bins for categorical analysis

### Advanced Modeling
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Ensemble methods (Voting, Stacking)
- [ ] Neural networks with TensorFlow/PyTorch
- [ ] Cross-validation with different strategies

### Visualization Enhancements
- [ ] Interactive plots with Plotly
- [ ] Animated visualizations
- [ ] Dashboard creation with Streamlit

## 📝 Key Insights Summary

1. **"Women and Children First"** - The maritime evacuation protocol was clearly followed
2. **Class Privilege** - Higher-class passengers had significantly better survival odds
3. **Economic Factor** - Ticket fare strongly correlated with survival chances
4. **Family Dynamics** - Moderate family sizes (2-4) had optimal survival rates
5. **Port Influence** - Embarkation port reflected passenger class distribution

## 🏆 Project Highlights

- **Complete EDA Pipeline:** From raw data to actionable insights
- **Visual Storytelling:** Clear, informative plots that tell the survival story
- **ML Implementation:** Multiple algorithms with proper evaluation
- **Real-world Application:** Historical disaster analysis with modern data science

## 📧 Contact

Feel free to reach out for questions or collaborations:
- **Email:** your.email@example.com
- **LinkedIn:** [Your LinkedIn Profile]
- **GitHub:** [Your GitHub Profile]

---

*This project demonstrates proficiency in data analysis, visualization, and machine learning using Python. The Titanic dataset serves as an excellent case study for understanding how socioeconomic factors influenced survival during historical disasters.*