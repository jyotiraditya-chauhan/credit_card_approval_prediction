# Credit Card Fraud Detection System 💳

A comprehensive machine learning-based system for detecting credit card fraud transactions using advanced analytics and multiple classification algorithms.

## 🎯 Project Overview

This project implements a complete credit card fraud detection pipeline that analyzes transaction patterns and predicts fraudulent activities using machine learning models. The system provides data-driven insights for smarter financial security decisions and focuses on key factors affecting fraud outcomes.

## ✨ Features

- **🔍 Comprehensive Data Analysis**: Explore transaction patterns, fraud distributions, and key risk factors
- **📊 Rich Visualizations**: 12+ interactive charts and graphs showing fraud patterns across different dimensions
- **🤖 Multiple ML Models**: Random Forest, Logistic Regression, and SVM classifiers with balanced class handling
- **📈 Performance Metrics**: Accuracy, AUC-ROC, Precision, Recall, F1-Score, and Confusion Matrix analysis
- **🔮 Real-time Prediction**: Single transaction fraud prediction with probability scores
- **💾 Data Management**: Automatic dummy data generation if no dataset is provided
- **🖥️ Interactive CLI**: User-friendly command-line interface for all operations

## 🏗️ System Architecture

The system consists of two main classes:

### CreditCardAnalyzer
Core analysis engine that handles:
- Data loading and preprocessing
- Exploratory data analysis
- Machine learning model training
- Model evaluation and comparison
- Individual transaction prediction

### CreditCardCLI
Command-line interface providing:
- Interactive menu system
- User input handling
- Result visualization
- Transaction checking interface

## 📋 Dataset Features

The system analyzes 11 key transaction features:

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| `TransactionAmount` | Transaction value in dollars | Float | $1 - $10,000 |
| `AccountAge` | Account age in days | Integer | 1 - 3,650 days |
| `NumTransactionsToday` | Daily transaction count | Integer | 0 - 50 |
| `AvgTransactionAmount` | Average transaction amount | Float | $5 - $5,000 |
| `TimeSinceLastTransaction` | Hours since last transaction | Float | 0.1 - 168 hours |
| `MerchantCategory` | Merchant type category | Integer | 1 - 10 |
| `TransactionHour` | Hour of transaction (24h format) | Integer | 0 - 23 |
| `DayOfWeek` | Day of the week | Integer | 1 - 7 |
| `IsWeekend` | Weekend flag | Binary | 0 or 1 |
| `CardType` | Card type identifier | Integer | 1 - 4 |
| `CVVMatch` | CVV verification status | Binary | 0 or 1 |

**Target Variable**: `IsFraud` (0 = Legitimate, 1 = Fraudulent)

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.7+
```

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Alternative Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Running the System
```bash
python credit_card_analyzer.py
```

### Main Menu Options

```
1. Load Data from CSV        - Import your transaction dataset
2. Explore Data             - Statistical analysis and insights
3. Visualize Data           - Generate comprehensive charts
4. Train Models             - Train ML models on your data
5. Evaluate Models          - Compare model performance
6. Check Single Transaction - Predict fraud for new transactions
0. Exit                     - Close the application
```

### Data Format

#### CSV Input Format
Your CSV file should contain comma-separated values without headers:
```csv
150.50,365,2,75.25,1.5,3,14,3,0,1,1,0
2500.00,45,8,300.00,0.2,1,2,1,1,2,1,1
75.25,1200,1,85.50,12.5,5,16,5,0,3,1,0
```

#### Single Transaction Input
When checking individual transactions, provide 11 comma-separated values:
```
150.50,365,2,75.25,1.5,3,14,3,0,1,1
```

## 🔧 System Workflow

### 1. Data Loading
- **Option A**: Load from CSV file (specify path or use default 'credit_card_data.csv')
- **Option B**: Auto-generate realistic dummy data (10,000 transactions with ~3% fraud rate)

### 2. Data Exploration
Provides comprehensive analysis including:
- Dataset shape and memory usage
- Missing value detection
- Fraud distribution statistics
- Transaction amount analysis
- Time-based patterns
- Feature correlations with fraud

### 3. Data Visualization
Generates 12 different visualizations:
- Fraud distribution pie chart
- Transaction amount histograms
- Hourly fraud patterns
- Account age distributions
- Daily transaction analysis
- CVV match patterns
- Merchant category risks
- Feature correlation heatmap
- Time-based analysis
- Weekend vs weekday patterns
- Card type analysis
- Scatter plots

### 4. Model Training
Trains three optimized models:

#### Random Forest Classifier
- 100 estimators
- Balanced class weights
- Handles feature importance

#### Logistic Regression
- Balanced class weights
- Maximum 1000 iterations
- Scaled features

#### Support Vector Machine (SVM)
- Probability estimation enabled
- Balanced class weights
- RBF kernel

### 5. Model Evaluation
Comprehensive performance analysis:
- **Accuracy Score**: Overall correctness
- **AUC-ROC Score**: Discrimination ability
- **Classification Report**: Precision, Recall, F1-Score
- **Confusion Matrix**: True/False Positives/Negatives
- **ROC Curves**: Visual performance comparison
- **Feature Importance**: For Random Forest model

### 6. Fraud Prediction
Individual transaction assessment:
- Binary prediction (Fraudulent/Legitimate)
- Fraud probability percentage
- Risk level classification:
  - **LOW RISK**: < 30% probability
  - **MEDIUM RISK**: 30-50% probability
  - **HIGH RISK**: > 50% probability

## 📊 Key Performance Indicators

### Fraud Detection Patterns
- **Peak fraud hours**: Typically 2-4 AM
- **High-risk amounts**: > $1,000 transactions
- **Account age factor**: New accounts (< 30 days) higher risk
- **Transaction frequency**: > 10 transactions/day suspicious
- **CVV mismatch**: Strong fraud indicator

### Model Performance Targets
- **Accuracy**: > 85%
- **AUC-ROC**: > 0.90
- **Precision**: Minimize false positives
- **Recall**: Catch maximum fraud cases

## 🛡️ Security Features

- **Balanced Learning**: Handles imbalanced fraud datasets
- **Feature Scaling**: Standardizes numerical features
- **Cross-validation**: Stratified train-test split maintains fraud ratios
- **Multiple Models**: Ensemble approach for robust predictions
- **Real-time Processing**: Single transaction evaluation

## 📁 File Structure

```
credit-card-fraud-detection/
│
├── credit_card_analyzer.py    # Main application file
├── credit_card_data.csv       # Sample dataset (auto-generated)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── examples/
    ├── sample_data.csv        # Example dataset
    └── transaction_examples.txt # Sample transactions
```

## 🔧 Configuration Options

### Data Generation Parameters
```python
n_samples = 10000              # Number of transactions to generate
fraud_rate = 0.03              # Target fraud percentage (3%)
random_seed = 42               # For reproducible results
```

### Model Parameters
```python
# Random Forest
n_estimators = 100
class_weight = 'balanced'

# Logistic Regression
max_iter = 1000
class_weight = 'balanced'

# SVM
kernel = 'rbf'
probability = True
class_weight = 'balanced'
```

## 📈 Example Output

### Fraud Analysis Summary
```
💳 Dataset shape: (10000, 12)
🚨 Fraud Distribution:
Legitimate (0): 9701 (97.01%)
Fraudulent (1): 299 (2.99%)

💰 Transaction Amount Analysis:
Average transaction: $287.45
Fraud avg amount: $1,245.67
Legitimate avg amount: $265.32

🕐 Peak fraud hour: 3:00 (8.45% fraud rate)
```

### Model Performance Example
```
🔍 Random Forest Results:
Accuracy: 0.9875
AUC Score: 0.9923

Classification Report:
              precision    recall  f1-score
Legitimate       0.99      1.00      0.99
Fraudulent       0.92      0.85      0.88
```

## 🐛 Troubleshooting

### Common Issues

1. **File Not Found Error**
   - System automatically generates dummy data if CSV not found
   - Ensure file path is correct

2. **Memory Issues**
   - Reduce dataset size for large files
   - Close visualization windows after viewing

3. **Model Training Failures**
   - Check data preprocessing completion
   - Ensure sufficient fraud samples in dataset

4. **Visualization Problems**
   - Install matplotlib and seaborn
   - Update display drivers if needed

## 🔮 Future Enhancements

- **Advanced Models**: Deep learning, ensemble methods
- **Real-time API**: REST API for production deployment
- **Database Integration**: PostgreSQL, MongoDB support
- **Advanced Features**: Geolocation, merchant history
- **Web Interface**: Dashboard for non-technical users
- **Alert System**: Email/SMS notifications for high-risk transactions


## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

For support, questions, or suggestions:
- Create an issue on GitHub
- Email: adityachuahan0369@gmail.com
- Documentation: [Project Wiki](https://github.com/jyotiraditya-chauhan/credit-card-fraud-detection/)

## 🙏 Acknowledgments

- Scikit-learn for machine learning algorithms
- Pandas for data manipulation
- Matplotlib/Seaborn for visualizations
- Credit card industry for fraud pattern insights

---

**⚠️ Disclaimer**: This system is for educational and research purposes. Always validate results with domain experts before using in production financial systems.
