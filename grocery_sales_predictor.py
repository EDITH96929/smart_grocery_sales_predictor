import pandas as pd
import numpy as np
import joblib
import warnings
import os
from datetime import datetime, timedelta
import argparse

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# --- Constants & Paths ---
SALES_CSV = './sales_data/sales_data.csv'
CALENDAR_CSV = './sales_data/calendar_data_full.csv'
MODEL_PATH = './sales_data/best_grocery_model.pkl'
GROWTH_RATE = 0.02  # 2% yearly price increase

class SmartGroceryPredictor:
    def __init__(self, sales_path=SALES_CSV, calendar_path=CALENDAR_CSV):
        self.sales_path = sales_path
        self.calendar_path = calendar_path
        self.model = None
        self.features = None
        self.df = None
        self.calendar_df = None
        self.sales_df = None
        
    def check_files(self):
        """Check if required data files exist"""
        missing_files = []
        if not os.path.exists(self.sales_path):
            missing_files.append(self.sales_path)
        if not os.path.exists(self.calendar_path):
            missing_files.append(self.calendar_path)
        
        if missing_files:
            print("❌ Missing files:")
            for file in missing_files:
                print(f"   - {file}")
            print("\n💡 Please ensure these files exist in the correct directory.")
            return False
        return True
    
    def inspect_data_files(self):
        """Inspect the structure of data files for debugging"""
        print("🔍 Inspecting data files...")
        
        try:
            # Check sales file
            print(f"\n📊 Sales file: {self.sales_path}")
            sales_sample = pd.read_csv(self.sales_path, nrows=5)
            print(f"   Columns: {list(sales_sample.columns)}")
            print(f"   Shape: {sales_sample.shape}")
            print(f"   Sample data:")
            print(sales_sample.to_string())
            
            if 'date' in sales_sample.columns:
                print(f"   Date column type: {sales_sample['date'].dtype}")
                print(f"   Sample dates: {sales_sample['date'].head(3).tolist()}")
            
        except Exception as e:
            print(f"   ❌ Error reading sales file: {str(e)}")
        
        try:
            # Check calendar file
            print(f"\n📅 Calendar file: {self.calendar_path}")
            calendar_sample = pd.read_csv(self.calendar_path, nrows=5)
            print(f"   Columns: {list(calendar_sample.columns)}")
            print(f"   Shape: {calendar_sample.shape}")
            print(f"   Sample data:")
            print(calendar_sample.to_string())
            
            if 'date' in calendar_sample.columns:
                print(f"   Date column type: {calendar_sample['date'].dtype}")
                print(f"   Sample dates: {calendar_sample['date'].head(3).tolist()}")
            
        except Exception as e:
            print(f"   ❌ Error reading calendar file: {str(e)}")
        
        print("\n" + "="*50)
    
    def load_data(self):
        """Load and merge sales and calendar data"""
        try:
            print("📁 Loading data...")
            
            # Load CSV files
            self.sales_df = pd.read_csv(self.sales_path)
            self.calendar_df = pd.read_csv(self.calendar_path)
            
            print(f"   Sales data shape: {self.sales_df.shape}")
            print(f"   Calendar data shape: {self.calendar_df.shape}")
            
            # Check if date columns exist
            if 'date' not in self.sales_df.columns:
                print("❌ 'date' column not found in sales data")
                print(f"   Available columns: {list(self.sales_df.columns)}")
                raise ValueError("'date' column not found in sales data")
            
            if 'date' not in self.calendar_df.columns:
                print("❌ 'date' column not found in calendar data")
                print(f"   Available columns: {list(self.calendar_df.columns)}")
                raise ValueError("'date' column not found in calendar data")
            
            # Convert date columns to datetime - handle different formats
            print("   Converting date columns...")
            
            # For sales data
            try:
                self.sales_df['date'] = pd.to_datetime(self.sales_df['date'])
                print(f"   ✅ Sales date converted: {self.sales_df['date'].dtype}")
            except Exception as e:
                print(f"   ❌ Error converting sales date: {str(e)}")
                # Try different date formats
                formats_to_try = [
                    '%d-%m-%Y',    # DD-MM-YYYY (like "13-01-2023")
                    '%Y-%m-%d',    # YYYY-MM-DD
                    '%m/%d/%Y',    # MM/DD/YYYY
                    '%d/%m/%Y',    # DD/MM/YYYY
                    '%m-%d-%Y',    # MM-DD-YYYY
                ]
                
                converted = False
                for fmt in formats_to_try:
                    try:
                        print(f"   Trying format: {fmt}")
                        self.sales_df['date'] = pd.to_datetime(self.sales_df['date'], format=fmt)
                        print(f"   ✅ Sales date converted with format {fmt}: {self.sales_df['date'].dtype}")
                        converted = True
                        break
                    except Exception as fmt_error:
                        print(f"   ❌ Format {fmt} failed: {str(fmt_error)[:100]}...")
                        continue
                
                if not converted:
                    # Last resort: use pandas' flexible parsing
                    try:
                        print("   Trying flexible parsing with dayfirst=True...")
                        self.sales_df['date'] = pd.to_datetime(self.sales_df['date'], dayfirst=True)
                        print(f"   ✅ Sales date converted with flexible parsing: {self.sales_df['date'].dtype}")
                    except Exception as final_error:
                        print(f"   ❌ All date parsing attempts failed: {str(final_error)}")
                        raise final_error
            
            # For calendar data
            try:
                self.calendar_df['date'] = pd.to_datetime(self.calendar_df['date'])
                print(f"   ✅ Calendar date converted: {self.calendar_df['date'].dtype}")
            except Exception as e:
                print(f"   ❌ Error converting calendar date: {str(e)}")
                # Try different date formats including DD-MM-YYYY
                formats_to_try = [
                    '%d-%m-%Y',    # DD-MM-YYYY (like "13-01-2023")
                    '%Y-%m-%d',    # YYYY-MM-DD
                    '%m/%d/%Y',    # MM/DD/YYYY
                    '%d/%m/%Y',    # DD/MM/YYYY
                    '%m-%d-%Y',    # MM-DD-YYYY
                ]
                
                converted = False
                for fmt in formats_to_try:
                    try:
                        print(f"   Trying format: {fmt}")
                        self.calendar_df['date'] = pd.to_datetime(self.calendar_df['date'], format=fmt)
                        print(f"   ✅ Calendar date converted with format {fmt}: {self.calendar_df['date'].dtype}")
                        converted = True
                        break
                    except Exception as fmt_error:
                        print(f"   ❌ Format {fmt} failed: {str(fmt_error)[:100]}...")
                        continue
                
                if not converted:
                    # Last resort: use pandas' flexible parsing
                    try:
                        print("   Trying flexible parsing with dayfirst=True...")
                        self.calendar_df['date'] = pd.to_datetime(self.calendar_df['date'], dayfirst=True)
                        print(f"   ✅ Calendar date converted with flexible parsing: {self.calendar_df['date'].dtype}")
                    except Exception as final_error:
                        print(f"   ❌ All date parsing attempts failed: {str(final_error)}")
                        raise final_error
            
            # Handle different column names
            if 'units_sold' in self.sales_df.columns:
                self.sales_df = self.sales_df.rename(columns={'units_sold': 'sales'})
                print("   ✅ Renamed 'units_sold' to 'sales'")
            
            # Check if sales column exists
            if 'sales' not in self.sales_df.columns:
                print("❌ 'sales' column not found in sales data")
                print(f"   Available columns: {list(self.sales_df.columns)}")
                raise ValueError("'sales' column not found. Please ensure your sales data has a 'sales' or 'units_sold' column")
            
            # Print date ranges for debugging
            print(f"   Sales date range: {self.sales_df['date'].min()} to {self.sales_df['date'].max()}")
            print(f"   Calendar date range: {self.calendar_df['date'].min()} to {self.calendar_df['date'].max()}")
            
            # Merge data
            print("   Merging datasets...")
            self.df = self.sales_df.merge(self.calendar_df, on='date', how='left')
            
            print(f"✅ Data loaded successfully! Shape: {self.df.shape}")
            print(f"   Columns: {list(self.df.columns)}")
            print(f"   Final date range: {self.df['date'].min()} to {self.df['date'].max()}")
            print(f"   Sample data:")
            print(self.df.head(3).to_string())
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            print("\n💡 Troubleshooting tips:")
            print("   1. Check if your CSV files exist in the './sales_data/' directory")
            print("   2. Ensure both files have a 'date' column")
            print("   3. Ensure sales data has 'sales' or 'units_sold' column")
            print("   4. Check date format in both files (should be YYYY-MM-DD or MM/DD/YYYY)")
            raise
    
    def preprocess(self):
        """Preprocess data and create features"""
        try:
            print("🔧 Preprocessing data...")
            df = self.df.copy()
            
            # Create datetime features
            df['dayofweek'] = df['date'].dt.weekday
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            
            # Handle categorical columns
            cat_cols = df.select_dtypes(['object', 'category']).columns.difference(['date'])
            print(f"   Categorical columns found: {list(cat_cols)}")
            
            for col in cat_cols:
                df[col] = df[col].fillna('None')
            
            # Create dummy variables
            if len(cat_cols) > 0:
                df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
            
            # Fill missing values
            df = df.fillna(0)
            
            # Separate features and target
            if 'sales' not in df.columns:
                raise ValueError("'sales' column not found in data")
            
            y = df['sales']
            X = df.drop(columns=['sales', 'date'])
            self.features = X.columns.tolist()
            
            print(f"✅ Preprocessing complete! Features: {X.shape[1]}, Samples: {X.shape[0]}")
            return X, y
            
        except Exception as e:
            print(f"❌ Error in preprocessing: {str(e)}")
            raise
    
    def train_models(self, X, y):
        """Train and tune models"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False, random_state=42
            )
            
            print(f"📊 Training models...")
            print(f"   Train set: {X_train.shape[0]} samples")
            print(f"   Test set: {X_test.shape[0]} samples")
            
            # Clean data
            X_train_clean = X_train.fillna(0)
            X_test_clean = X_test.fillna(0)
            
            # Train baseline models
            print("   Training baseline models...")
            lr = LinearRegression()
            lr.fit(X_train_clean, y_train)
            lr_pred = lr.predict(X_test_clean)
            
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train_clean, y_train)
            rf_pred = rf.predict(X_test_clean)
            
            # Calculate baseline errors
            lr_mae = mean_absolute_error(y_test, lr_pred)
            rf_mae = mean_absolute_error(y_test, rf_pred)
            
            print(f"   ✅ Linear Regression MAE: {lr_mae:.4f}")
            print(f"   ✅ Random Forest MAE: {rf_mae:.4f}")
            
            # Hyperparameter tuning
            print("   Tuning Random Forest hyperparameters...")
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
            
            grid = GridSearchCV(
                RandomForestRegressor(random_state=42),
                param_grid,
                cv=3,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            
            grid.fit(X_train_clean, y_train)
            self.model = grid.best_estimator_
            
            # Evaluate tuned model
            tuned_pred = self.model.predict(X_test_clean)
            tuned_mae = mean_absolute_error(y_test, tuned_pred)
            
            print(f"   ✅ Best parameters: {grid.best_params_}")
            print(f"   ✅ Tuned Random Forest MAE: {tuned_mae:.4f}")
            
            return {
                'lr_mae': lr_mae,
                'rf_mae': rf_mae,
                'tuned_mae': tuned_mae,
                'improvement': ((rf_mae - tuned_mae) / rf_mae) * 100
            }
            
        except Exception as e:
            print(f"❌ Error training models: {str(e)}")
            raise
    
    def save_model(self):
        """Save the trained model"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            
            # Save model and features
            joblib.dump({
                'model': self.model,
                'features': self.features
            }, MODEL_PATH)
            
            print(f"💾 Model saved to {MODEL_PATH}")
            
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            raise
    
    def load_model(self):
        """Load a previously trained model"""
        try:
            if os.path.exists(MODEL_PATH):
                saved_data = joblib.load(MODEL_PATH)
                self.model = saved_data['model']
                self.features = saved_data['features']
                print(f"📥 Model loaded from {MODEL_PATH}")
                return True
            else:
                print(f"⚠️  No saved model found at {MODEL_PATH}")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def predict_sales(self, start_date, end_date=None, category=None):
        """Make sales predictions for given date range"""
        try:
            if self.model is None:
                raise ValueError("Model not trained or loaded")
            
            # Handle single date prediction
            if end_date is None:
                end_date = start_date
            
            # Convert to datetime
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            print(f"🔮 Predicting sales from {start_date.date()} to {end_date.date()}")
            if category:
                print(f"   Category filter: {category}")
            
            # Create date range
            dates = pd.date_range(start_date, end_date)
            
            # Build features
            X_pred = self._build_features(dates, category)
            
            # Make predictions
            predictions = self.model.predict(X_pred)
            
            # Create results dataframe
            results = pd.DataFrame({
                'date': dates,
                'predicted_sales': predictions
            })
            
            # Add revenue predictions if possible
            results = self._add_revenue_predictions(results, category)
            
            print(f"✅ Predictions completed for {len(results)} days")
            return results
            
        except Exception as e:
            print(f"❌ Error making predictions: {str(e)}")
            raise
    
    def _build_features(self, dates, category=None):
        """Build feature matrix for prediction dates"""
        # Create basic datetime features
        feats = pd.DataFrame(index=dates)
        feats['dayofweek'] = feats.index.weekday
        feats['month'] = feats.index.month
        feats['day'] = feats.index.day
        
        # Add calendar features if available
        if self.calendar_df is not None and not self.calendar_df.empty:
            cal = self.calendar_df.set_index('date')
            cal_cat_cols = self.calendar_df.select_dtypes(['object', 'category']).columns.difference(['date'])
            
            for col in cal_cat_cols:
                cal_dummies = pd.get_dummies(self.calendar_df[col].fillna('None'), prefix=col, drop_first=True)
                cal_dummies.index = self.calendar_df['date']
                feats = feats.join(cal_dummies, how='left')
        
        # Ensure all required columns are present
        feats = feats.reindex(columns=self.features, fill_value=0)
        
        # Set category if specified
        if category and f"category_{category}" in feats.columns:
            feats[f"category_{category}"] = 1
        
        return feats.fillna(0)
    
    def _add_revenue_predictions(self, pred_df, category=None):
        """Add revenue predictions to the results"""
        try:
            prices = self.sales_df.copy()
            
            if category and 'category' in prices.columns:
                prices = prices[prices['category'] == category]
            
            # Check if unit_price column exists
            if 'unit_price' not in prices.columns:
                print("   ⚠️  'unit_price' column not found. Using default price of 1.0")
                pred_df['unit_price'] = 1.0
                pred_df['predicted_revenue'] = pred_df['predicted_sales'] * 1.0
                return pred_df
            
            price_lookup = prices.groupby('date')['unit_price'].mean()
            
            if len(price_lookup) == 0:
                print("   ⚠️  No price data found. Using default price of 1.0")
                pred_df['unit_price'] = 1.0
            else:
                last_date = price_lookup.index.max()
                last_price = price_lookup.loc[last_date]
                
                # Apply price lookup and growth
                years_diff = (pred_df['date'] - last_date).dt.days / 365.0
                base_prices = pred_df['date'].map(price_lookup).fillna(last_price)
                growth_factor = ((1 + GROWTH_RATE) ** years_diff.clip(lower=0))
                pred_df['unit_price'] = base_prices * growth_factor
            
            pred_df['predicted_revenue'] = pred_df['predicted_sales'] * pred_df['unit_price']
            return pred_df
            
        except Exception as e:
            print(f"   ⚠️  Error calculating revenue: {str(e)}")
            pred_df['unit_price'] = 1.0
            pred_df['predicted_revenue'] = pred_df['predicted_sales']
            return pred_df
    
    def plot_predictions(self, predictions, save_plot=False):
        """Plot prediction results"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Sales plot
            ax1.plot(predictions['date'], predictions['predicted_sales'], 
                    marker='o', linewidth=2, markersize=6)
            ax1.set_title('Predicted Sales', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Sales Units')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Revenue plot
            if 'predicted_revenue' in predictions.columns:
                ax2.bar(predictions['date'], predictions['predicted_revenue'], 
                       color='green', alpha=0.7)
                ax2.set_title('Predicted Revenue', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Revenue ($)')
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_plot:
                plot_path = './sales_data/predictions_plot.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"📊 Plot saved to {plot_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating plots: {str(e)}")
    
    def run_eda(self, save_plots=False):
        """Run exploratory data analysis"""
        try:
            print("📊 Running Exploratory Data Analysis...")
            
            if self.df is None:
                print("❌ No data loaded. Please run load_data() first.")
                return
            
            # Basic statistics
            print("\n=== Data Summary ===")
            print(f"Shape: {self.df.shape}")
            print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
            print(f"Total sales: {self.df['sales'].sum():,.0f}")
            print(f"Average daily sales: {self.df['sales'].mean():.2f}")
            print(f"Sales std: {self.df['sales'].std():.2f}")
            
            # Missing values
            print("\n=== Missing Values ===")
            missing = self.df.isnull().sum()
            print(missing[missing > 0])
            
            # Create visualizations
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Time series plot
            self.df.set_index('date')['sales'].plot(ax=axes[0,0], title='Sales Over Time')
            axes[0,0].grid(True, alpha=0.3)
            
            # Sales distribution
            axes[0,1].hist(self.df['sales'], bins=50, alpha=0.7, color='skyblue')
            axes[0,1].set_title('Sales Distribution')
            axes[0,1].grid(True, alpha=0.3)
            
            # Monthly seasonality
            monthly_sales = self.df.groupby(self.df['date'].dt.month)['sales'].mean()
            axes[1,0].bar(monthly_sales.index, monthly_sales.values, color='lightgreen')
            axes[1,0].set_title('Average Sales by Month')
            axes[1,0].set_xlabel('Month')
            axes[1,0].grid(True, alpha=0.3)
            
            # Day of week pattern
            dow_sales = self.df.groupby(self.df['date'].dt.dayofweek)['sales'].mean()
            axes[1,1].bar(dow_sales.index, dow_sales.values, color='orange')
            axes[1,1].set_title('Average Sales by Day of Week')
            axes[1,1].set_xlabel('Day of Week (0=Monday)')
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                eda_path = './sales_data/eda_plots.png'
                plt.savefig(eda_path, dpi=300, bbox_inches='tight')
                print(f"📊 EDA plots saved to {eda_path}")
            
            plt.show()
            
            print("✅ EDA completed!")
            
        except Exception as e:
            print(f"❌ Error in EDA: {str(e)}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Smart Grocery Sales Predictor')
    parser.add_argument('--mode', choices=['train', 'predict', 'eda', 'all', 'inspect'], 
                       default='all', help='Mode to run')
    parser.add_argument('--start-date', type=str, help='Start date for prediction (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for prediction (YYYY-MM-DD)')
    parser.add_argument('--category', type=str, help='Product category filter')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to files')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = SmartGroceryPredictor()
    
    print("🛒 Smart Grocery Sales Predictor")
    print("=" * 50)
    
    # Check files
    if not predictor.check_files():
        return
    
    # Inspect mode for debugging
    if args.mode == 'inspect':
        predictor.inspect_data_files()
        return
    
    try:
        if args.mode in ['train', 'all']:
            print("\n📚 TRAINING MODE")
            print("-" * 30)
            
            # Load and preprocess data
            predictor.load_data()
            X, y = predictor.preprocess()
            
            # Train models
            results = predictor.train_models(X, y)
            
            # Save model
            predictor.save_model()
            
            # Print summary
            print(f"\n🎯 TRAINING SUMMARY")
            print(f"   Linear Regression MAE: {results['lr_mae']:.4f}")
            print(f"   Random Forest MAE: {results['rf_mae']:.4f}")
            print(f"   Tuned RF MAE: {results['tuned_mae']:.4f}")
            print(f"   Improvement: {results['improvement']:.2f}%")
        
        if args.mode in ['eda', 'all']:
            print("\n📊 EXPLORATORY DATA ANALYSIS")
            print("-" * 30)
            
            if predictor.df is None:
                predictor.load_data()
            
            predictor.run_eda(save_plots=args.save_plots)
        
        if args.mode in ['predict', 'all']:
            print("\n🔮 PREDICTION MODE")
            print("-" * 30)
            
            # Load model if not already trained
            if predictor.model is None:
                if not predictor.load_model():
                    print("❌ No trained model found. Please run training first.")
                    return
            
            # Load data if not already loaded
            if predictor.df is None:
                predictor.load_data()
            
            # Set default prediction dates
            if args.start_date:
                start_date = args.start_date
            else:
                # Predict for next 7 days after last data point
                last_date = predictor.calendar_df['date'].max()
                start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            if args.end_date:
                end_date = args.end_date
            else:
                # 7 days from start date
                end_date = (pd.to_datetime(start_date) + timedelta(days=6)).strftime('%Y-%m-%d')
            
            # Make predictions
            predictions = predictor.predict_sales(start_date, end_date, args.category)
            
            # Display results
            print("\n📋 PREDICTION RESULTS")
            print(predictions.to_string(index=False))
            
            # Summary statistics
            print(f"\n📈 SUMMARY")
            print(f"   Total predicted sales: {predictions['predicted_sales'].sum():.0f} units")
            print(f"   Average daily sales: {predictions['predicted_sales'].mean():.2f} units")
            if 'predicted_revenue' in predictions.columns:
                print(f"   Total predicted revenue: ${predictions['predicted_revenue'].sum():.2f}")
                print(f"   Average daily revenue: ${predictions['predicted_revenue'].mean():.2f}")
            
            # Plot results
            predictor.plot_predictions(predictions, save_plot=args.save_plots)
        
        print("\n✅ All operations completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Operation interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()