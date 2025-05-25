import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

class CreditCardAnalyzer:
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        # Define column names for the credit card fraud dataset
        self.column_names = [
            'TransactionAmount', 'AccountAge', 'NumTransactionsToday', 'AvgTransactionAmount',
            'TimeSinceLastTransaction', 'MerchantCategory', 'TransactionHour', 
            'DayOfWeek', 'IsWeekend', 'CardType', 'CVVMatch', 'IsFraud'
        ]
        
    def load_data(self, file_path='credit_card_data.csv'):
        """Load data from CSV file"""
        try:
            # Load data with proper column names
            self.df = pd.read_csv(file_path, header=None, names=self.column_names)
            print(f"✅ Loaded data from {file_path}")
            
            # Basic data validation
            if self.df.empty:
                raise ValueError("Dataset is empty")
            
            print(f"💳 Dataset shape: {self.df.shape}")
            return True
            
        except FileNotFoundError:
            print(f"❌ File {file_path} not found. Generating dummy data instead...")
            return self.generate_dummy_data()
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            print("Generating dummy data instead...")
            return self.generate_dummy_data()
    
    def generate_dummy_data(self, n_samples=10000):
        """Generate realistic dummy credit card transaction data"""
        print("🔄 Generating realistic dummy credit card transaction data...")
        
        np.random.seed(42)
        
        # Generate features for credit card transactions
        data = {
            'TransactionAmount': np.random.lognormal(3, 1.5, n_samples),  # Log-normal for realistic amounts
            'AccountAge': np.random.gamma(2, 500, n_samples),  # Account age in days
            'NumTransactionsToday': np.random.poisson(3, n_samples),  # Number of transactions today
            'AvgTransactionAmount': np.random.lognormal(3.5, 1, n_samples),  # Average transaction amount
            'TimeSinceLastTransaction': np.random.exponential(2, n_samples),  # Hours since last transaction
            'MerchantCategory': np.random.randint(1, 11, n_samples),  # 1-10 merchant categories
            'TransactionHour': np.random.randint(0, 24, n_samples),  # Hour of day (0-23)
            'DayOfWeek': np.random.randint(1, 8, n_samples),  # Day of week (1-7)
            'IsWeekend': np.random.binomial(1, 0.28, n_samples),  # Weekend flag
            'CardType': np.random.randint(1, 5, n_samples),  # 1-4 card types (Visa, MC, Amex, etc.)
            'CVVMatch': np.random.binomial(1, 0.98, n_samples)  # CVV match (98% success rate)
        }
        
        # Ensure realistic bounds
        data['TransactionAmount'] = np.clip(data['TransactionAmount'], 1, 10000)
        data['AccountAge'] = np.clip(data['AccountAge'], 1, 3650).astype(int)  # Max 10 years
        data['NumTransactionsToday'] = np.clip(data['NumTransactionsToday'], 0, 50)
        data['AvgTransactionAmount'] = np.clip(data['AvgTransactionAmount'], 5, 5000)
        data['TimeSinceLastTransaction'] = np.clip(data['TimeSinceLastTransaction'], 0.1, 168)  # Max 1 week
        
        # Create fraud outcome based on risk factors
        fraud_score = (
            0.3 * (data['TransactionAmount'] > 1000).astype(int) +  # High amount
            0.2 * (data['NumTransactionsToday'] > 10).astype(int) +  # Many transactions today
            0.15 * (data['TimeSinceLastTransaction'] < 0.5).astype(int) +  # Very recent transaction
            0.1 * (data['TransactionHour'] < 6).astype(int) +  # Late night/early morning
            0.1 * (1 - data['CVVMatch']) +  # CVV mismatch
            0.05 * (data['AccountAge'] < 30).astype(int) +  # New account
            0.1 * np.random.random(n_samples)  # Random component
        )
        
        # Create binary fraud outcome (approximately 3% fraud rate)
        data['IsFraud'] = (fraud_score > 0.25).astype(int)
        
        self.df = pd.DataFrame(data)
        self.df = self.df.round(2)  # Round to 2 decimal places
        print(f"✅ Generated {n_samples} transactions with {self.df['IsFraud'].mean():.2%} fraud rate")
        
        # Save to CSV for future use
        self.save_dummy_data()
        return True
    
    def save_dummy_data(self):
        """Save generated dummy data to CSV file"""
        try:
            self.df.to_csv('credit_card_data.csv', index=False, header=False)
            print("💾 Dummy data saved to 'credit_card_data.csv'")
        except Exception as e:
            print(f"⚠️ Could not save dummy data: {str(e)}")
    
    def explore_data(self): 
        """Perform exploratory data analysis"""
        if self.df is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        print("\n" + "="*60)
        print("🔍 CREDIT CARD FRAUD ANALYSIS")
        print("="*60)
        
        # Basic info
        print("\n📋 Dataset Info:")
        print(f"Shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values
        print("\n🔍 Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("✅ No missing values found")
        else:
            print(missing[missing > 0])
        
        # Fraud distribution
        print("\n🚨 Fraud Distribution:")
        fraud_dist = self.df['IsFraud'].value_counts()
        print(f"Legitimate (0): {fraud_dist[0]} ({fraud_dist[0]/len(self.df):.2%})")
        print(f"Fraudulent (1): {fraud_dist[1]} ({fraud_dist[1]/len(self.df):.2%})")
        
        # Transaction amount statistics
        print("\n💰 Transaction Amount Analysis:")
        print(f"Average transaction: ${self.df['TransactionAmount'].mean():.2f}")
        print(f"Median transaction: ${self.df['TransactionAmount'].median():.2f}")
        print(f"Max transaction: ${self.df['TransactionAmount'].max():.2f}")
        print(f"Fraud avg amount: ${self.df[self.df['IsFraud']==1]['TransactionAmount'].mean():.2f}")
        print(f"Legitimate avg amount: ${self.df[self.df['IsFraud']==0]['TransactionAmount'].mean():.2f}")
        
        # Time-based analysis
        print("\n🕐 Time-based Analysis:")
        fraud_by_hour = self.df.groupby('TransactionHour')['IsFraud'].mean()
        peak_fraud_hour = fraud_by_hour.idxmax()
        print(f"Peak fraud hour: {peak_fraud_hour}:00 ({fraud_by_hour[peak_fraud_hour]:.2%} fraud rate)")
        
        # Statistical summary
        print("\n📊 Statistical Summary:")
        print(self.df.describe())
        
        # Feature correlations with fraud
        print("\n🔗 Feature Correlations with Fraud:")
        correlations = self.df.corr()['IsFraud'].drop('IsFraud').sort_values(key=abs, ascending=False)
        for feature, corr in correlations.items():
            print(f"{feature}: {corr:.3f}")
    
    def visualize_data(self):
        """Create comprehensive data visualizations"""
        if self.df is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        print("\n📈 Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Fraud distribution
        plt.subplot(3, 4, 1)
        fraud_counts = self.df['IsFraud'].value_counts()
        colors = ['#2ecc71', '#e74c3c']  # Green for legitimate, red for fraud
        plt.pie(fraud_counts.values, labels=['Legitimate', 'Fraudulent'], 
                autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Fraud Distribution', fontsize=12, fontweight='bold')
        
        # 2. Transaction amount distribution
        plt.subplot(3, 4, 2)
        legitimate = self.df[self.df['IsFraud'] == 0]['TransactionAmount']
        fraudulent = self.df[self.df['IsFraud'] == 1]['TransactionAmount']
        plt.hist([legitimate, fraudulent], bins=50, alpha=0.7, 
                label=['Legitimate', 'Fraudulent'], color=['#2ecc71', '#e74c3c'])
        plt.xlabel('Transaction Amount ($)')
        plt.ylabel('Frequency')
        plt.title('Transaction Amount Distribution')
        plt.legend()
        plt.xlim(0, 2000)  # Focus on lower amounts for visibility
        
        # 3. Fraud by hour of day
        plt.subplot(3, 4, 3)
        fraud_by_hour = self.df.groupby('TransactionHour')['IsFraud'].mean()
        plt.bar(fraud_by_hour.index, fraud_by_hour.values, color='#3498db', alpha=0.7)
        plt.xlabel('Hour of Day')
        plt.ylabel('Fraud Rate')
        plt.title('Fraud Rate by Hour')
        plt.xticks(range(0, 24, 4))
        
        # 4. Account age vs fraud
        plt.subplot(3, 4, 4)
        plt.boxplot([self.df[self.df['IsFraud'] == 0]['AccountAge'],
                    self.df[self.df['IsFraud'] == 1]['AccountAge']], 
                   labels=['Legitimate', 'Fraudulent'])
        plt.ylabel('Account Age (days)')
        plt.title('Account Age by Transaction Type')
        
        # 5. Number of transactions today
        plt.subplot(3, 4, 5)
        trans_today = pd.crosstab(self.df['NumTransactionsToday'], self.df['IsFraud'])
        trans_today.plot(kind='bar', ax=plt.gca(), color=['#2ecc71', '#e74c3c'])
        plt.title('Daily Transaction Count')
        plt.xlabel('Number of Transactions Today')
        plt.ylabel('Count')
        plt.legend(['Legitimate', 'Fraudulent'])
        plt.xticks(rotation=45)
        
        # 6. CVV match analysis
        plt.subplot(3, 4, 6)
        cvv_fraud = pd.crosstab(self.df['CVVMatch'], self.df['IsFraud'])
        cvv_fraud.plot(kind='bar', ax=plt.gca(), color=['#2ecc71', '#e74c3c'])
        plt.title('CVV Match vs Fraud')
        plt.xlabel('CVV Match (0=No, 1=Yes)')
        plt.ylabel('Count')
        plt.legend(['Legitimate', 'Fraudulent'])
        plt.xticks(rotation=0)
        
        # 7. Merchant category analysis
        plt.subplot(3, 4, 7)
        merchant_fraud = self.df.groupby('MerchantCategory')['IsFraud'].mean()
        plt.bar(merchant_fraud.index, merchant_fraud.values, color='#9b59b6', alpha=0.7)
        plt.xlabel('Merchant Category')
        plt.ylabel('Fraud Rate')
        plt.title('Fraud Rate by Merchant Category')
        
        # 8. Correlation heatmap
        plt.subplot(3, 4, 8)
        corr_matrix = self.df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=plt.gca(), cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix')
        
        # 9. Time since last transaction
        plt.subplot(3, 4, 9)
        plt.hist([self.df[self.df['IsFraud'] == 0]['TimeSinceLastTransaction'],
                 self.df[self.df['IsFraud'] == 1]['TimeSinceLastTransaction']], 
                bins=30, alpha=0.7, label=['Legitimate', 'Fraudulent'],
                color=['#2ecc71', '#e74c3c'])
        plt.xlabel('Hours Since Last Transaction')
        plt.ylabel('Frequency')
        plt.title('Time Since Last Transaction')
        plt.legend()
        
        # 10. Weekend vs weekday fraud
        plt.subplot(3, 4, 10)
        weekend_fraud = pd.crosstab(self.df['IsWeekend'], self.df['IsFraud'])
        weekend_fraud.plot(kind='bar', ax=plt.gca(), color=['#2ecc71', '#e74c3c'])
        plt.title('Weekend vs Weekday Fraud')
        plt.xlabel('Is Weekend (0=No, 1=Yes)')
        plt.ylabel('Count')
        plt.legend(['Legitimate', 'Fraudulent'])
        plt.xticks(rotation=0)
        
        # 11. Card type analysis
        plt.subplot(3, 4, 11)
        card_fraud = self.df.groupby('CardType')['IsFraud'].mean()
        plt.bar(card_fraud.index, card_fraud.values, color='#f39c12', alpha=0.7)
        plt.xlabel('Card Type')
        plt.ylabel('Fraud Rate')
        plt.title('Fraud Rate by Card Type')
        
        # 12. Transaction amount vs account age (scatter)
        plt.subplot(3, 4, 12)
        colors_scatter = ['#2ecc71' if x == 0 else '#e74c3c' for x in self.df['IsFraud']]
        plt.scatter(self.df['AccountAge'], self.df['TransactionAmount'], 
                   c=colors_scatter, alpha=0.5, s=10)
        plt.xlabel('Account Age (days)')
        plt.ylabel('Transaction Amount ($)')
        plt.title('Account Age vs Transaction Amount')
        plt.ylim(0, 2000)  # Limit y-axis for better visibility
        
        plt.tight_layout()
        plt.show()
        print("✅ Visualizations created successfully!")
    
    def preprocess_data(self):
        """Preprocess data for machine learning"""
        if self.df is None:
            print("❌ No data loaded. Please load data first.")
            return False
        
        print("\n🔄 Preprocessing data...")
        
        try:
            # Create a copy for preprocessing
            df_processed = self.df.copy()
            
            # Handle any missing values (fill with median/mode)
            for col in df_processed.columns:
                if df_processed[col].dtype in ['int64', 'float64']:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                else:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
            
            # Separate features and target
            X = df_processed.drop('IsFraud', axis=1)
            y = df_processed['IsFraud']
            
            self.feature_names = X.columns.tolist()
            
            # Split the data (stratified to maintain fraud ratio)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale the features
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            print(f"✅ Data preprocessed successfully!")
            print(f"Training set: {self.X_train.shape[0]} samples")
            print(f"Test set: {self.X_test.shape[0]} samples")
            print(f"Features: {len(self.feature_names)}")
            print(f"Training fraud rate: {self.y_train.mean():.2%}")
            print(f"Test fraud rate: {self.y_test.mean():.2%}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error preprocessing data: {str(e)}")
            return False
    
    def train_models(self):
        """Train multiple machine learning models"""
        if self.X_train is None:
            print("❌ Data not preprocessed. Please preprocess data first.")
            return False
        
        print("\n🤖 Training machine learning models...")
        
        # Define models with parameters suitable for imbalanced data
        models_config = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                class_weight='balanced',  # Handle imbalanced data
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                class_weight='balanced',
                random_state=42, 
                max_iter=1000
            ),
            'SVM': SVC(
                probability=True, 
                class_weight='balanced',
                random_state=42
            )
        }
        
        # Train models
        for name, model in models_config.items():
            print(f"Training {name}...")
            try:
                if name == 'Random Forest':
                    model.fit(self.X_train, self.y_train)
                else:
                    model.fit(self.X_train_scaled, self.y_train)
                
                self.models[name] = model
                print(f"✅ {name} trained successfully!")
                
            except Exception as e:
                print(f"❌ Error training {name}: {str(e)}")
        
        return len(self.models) > 0
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        if not self.models:
            print("❌ No models trained. Please train models first.")
            return
        
        print("\n" + "="*60)
        print("📊 FRAUD DETECTION MODEL EVALUATION")
        print("="*60)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n🔍 {name} Results:")
            print("-" * 30)
            
            try:
                # Make predictions
                if name == 'Random Forest':
                    y_pred = model.predict(self.X_test)
                    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                else:
                    y_pred = model.predict(self.X_test_scaled)
                    y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                auc_score = roc_auc_score(self.y_test, y_pred_proba)
                
                results[name] = {
                    'accuracy': accuracy,
                    'auc': auc_score,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"Accuracy: {accuracy:.4f}")
                print(f"AUC Score: {auc_score:.4f}")
                
                print("\nClassification Report:")
                print(classification_report(self.y_test, y_pred, 
                                           target_names=['Legitimate', 'Fraudulent']))
                
                print("Confusion Matrix:")
                cm = confusion_matrix(self.y_test, y_pred)
                print(f"True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
                print(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")
                
            except Exception as e:
                print(f"❌ Error evaluating {name}: {str(e)}")
        
        # Plot ROC curves
        self._plot_roc_curves(results)
        
        # Feature importance for Random Forest
        if 'Random Forest' in self.models:
            self._plot_feature_importance()
        
        return results
    
    def _plot_roc_curves(self, results):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for name, result in results.items():
            if 'probabilities' in result:
                fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
                plt.plot(fpr, tpr, linewidth=2, 
                        label=f"{name} (AUC = {result['auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Fraud Detection Models', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def _plot_feature_importance(self):
        """Plot feature importance for Random Forest"""
        if 'Random Forest' not in self.models:
            return
        
        model = self.models['Random Forest']
        importance = model.feature_importances_
        
        # Create DataFrame for sorting
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(feature_importance)), feature_importance['importance'], 
                color='#3498db', edgecolor='navy', alpha=0.7)
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title('Feature Importance - Fraud Detection (Random Forest)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()
    
    def predict_single(self, input_data):
        """Make fraud prediction for a single transaction"""
        if not self.models:
            return None, "No models trained"
        
        try:
            # Use the best performing model (Random Forest by default)
            model = self.models.get('Random Forest', list(self.models.values())[0])
            
            # Create input DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Ensure all features are present
            for feature in self.feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            # Reorder columns to match training data
            input_df = input_df[self.feature_names]
            
            # Make prediction
            if 'Random Forest' in self.models:
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]
            else:
                input_scaled = self.scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1]
            
            return prediction, probability
            
        except Exception as e:
            return None, f"Error making prediction: {str(e)}"


class CreditCardCLI:
    def __init__(self, analyzer):
        self.analyzer = analyzer
    
    def run(self):
        """Run the CLI interface"""
        print("\n" + "="*60)
        print("💳 CREDIT CARD FRAUD DETECTION & ANALYSIS SYSTEM")
        print("="*60)
        
        while True:
            print("\n📋 Main Menu:")
            print("1. Load Data from CSV")
            print("2. Explore Data")
            print("3. Visualize Data")
            print("4. Train Models")
            print("5. Evaluate Models")
            print("6. Check Single Transaction")
            print("0. Exit")
            
            try:
                choice = input("\nSelect an option (0-6): ").strip()
                
                if choice == '0':
                    print("👋 Thank you for using the Credit Card Fraud Detection System!")
                    break
                elif choice == '1':
                    self.load_data_cli()
                elif choice == '2':
                    self.analyzer.explore_data()
                elif choice == '3':
                    self.analyzer.visualize_data()
                elif choice == '4':
                    self.train_models_cli()
                elif choice == '5':
                    self.analyzer.evaluate_models()
                elif choice == '6':
                    self.check_transaction_cli()
                else:
                    print("❌ Invalid option. Please try again.")
                    
                input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def load_data_cli(self):
        """Load data via CLI"""
        file_path = input("Enter CSV file path (or press Enter for 'credit_card_data.csv'): ").strip()
        if not file_path:
            file_path = 'credit_card_data.csv'
        
        self.analyzer.load_data(file_path)
    
    def train_models_cli(self):
        """Train models via CLI"""
        if self.analyzer.df is None:
            print("❌ Please load data first!")
            return
        
        if self.analyzer.preprocess_data():
            self.analyzer.train_models()
        else:
            print("❌ Failed to preprocess data")
    
    def check_transaction_cli(self):
        """Check transaction for fraud via CLI"""
        if not self.analyzer.models:
            print("❌ Please train models first!")
            return
        
        print("\n💳 Enter Transaction Information:")
        print("Format: TransactionAmount,AccountAge,NumTransactionsToday,AvgTransactionAmount,")
        print("        TimeSinceLastTransaction,MerchantCategory,TransactionHour,DayOfWeek,")
        print("        IsWeekend,CardType,CVVMatch")
        print("Example: 150.50,365,2,75.25,1.5,3,14,3,0,1,1")
        print("\nField explanations:")
        print("- TransactionAmount: Amount in dollars (e.g., 150.50)")
        print("- AccountAge: Age in days (e.g., 365)")
        print("- NumTransactionsToday: Count (e.g., 2)")
        print("- AvgTransactionAmount: Average amount in dollars (e.g., 75.25)")
        print("- TimeSinceLastTransaction: Hours (e.g., 1.5)")
        print("- MerchantCategory: 1-10 (e.g., 3)")
        print("- TransactionHour: 0-23 (e.g., 14)")
        print("- DayOfWeek: 1-7 (e.g., 3)")
        print("- IsWeekend: 0 or 1 (e.g., 0)")
        print("- CardType: 1-4 (e.g., 1)")
        print("- CVVMatch: 0 or 1 (e.g., 1)")
        
        try:
            user_input = input("\nEnter values (comma-separated): ").strip()
            values = [float(x.strip()) for x in user_input.split(',')]
            
            if len(values) != 11:
                print("❌ Please enter exactly 11 values")
                return
            
            input_data = dict(zip(self.analyzer.feature_names, values))
            
            # Make prediction
            prediction, probability = self.analyzer.predict_single(input_data)
            
            if prediction is not None:
                print("\n" + "="*50)
                print("🔮 FRAUD DETECTION RESULT")
                print("="*50)
                
                print("Transaction Information:")
                for feature, value in input_data.items():
                    print(f"{feature}: {value}")
                
                print(f"\nPrediction: {'🚨 FRAUDULENT' if prediction == 1 else '✅ LEGITIMATE'}")
                print(f"Fraud Probability: {probability:.2%}")
                
                if prediction == 1:
                    print("⚠️  HIGH RISK - Transaction flagged for review")
                    print("   Recommended actions:")
                    print("   • Verify transaction with cardholder")
                    print("   • Check for suspicious patterns")
                    print("   • Consider blocking transaction")
                else:
                    if probability > 0.3:
                        print("⚡ MEDIUM RISK - Monitor transaction")
                    else:
                        print("✅ LOW RISK - Transaction appears legitimate")
                
                print("="*50)
            else:
                print(f"❌ Prediction failed: {probability}")
                
        except ValueError:
           print("❌ Please enter valid numerical values")
        except Exception as e:
           print(f"❌ Error: {str(e)}")


def main():
   """Main function"""
   print("💳 Credit Card Fraud Detection & Analysis System")
   print("🔐 Protecting your transactions with machine learning")
   
   try:
       # Create analyzer instance
       analyzer = CreditCardAnalyzer()
       
       # Start CLI interface
       cli = CreditCardCLI(analyzer)
       cli.run()
           
   except KeyboardInterrupt:
       print("\n👋 Goodbye!")
   except Exception as e:
       print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
   main()