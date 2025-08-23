# 🛒 Smart Grocery Sales Predictor

A Machine Learning project that predicts **grocery sales and revenue trends** using regression models, feature engineering, and interactive visualizations.  
It includes an interactive **Jupyter dashboard** powered by `ipywidgets`, allowing users to explore data (EDA) and forecast sales/revenue for specific categories and time ranges.  

---

## 🚀 Features
- 📊 **Exploratory Data Analysis (EDA)**: Interactive dropdowns to explore summary, time series, histograms, and seasonal patterns.  
- 🤖 **Machine Learning Models**: Linear Regression & Random Forest (with GridSearch hyperparameter tuning).  
- 🏆 **Performance Metrics**: Tracks model performance using **Mean Absolute Error (MAE)**.  
- 📈 **Revenue Forecasting**: Predicts future revenue considering price growth (default +2% yearly).  
- 🎛 **Interactive Dashboard**: Widgets for selecting categories, date ranges, and visualization options.  

---

## 🛠️ Tech Stack
- **Languages**: Python 🐍  
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **ML Models**: Linear Regression, Random Forest Regressor  
- **Dashboard**: ipywidgets + Jupyter Notebook  
- **Deployment Ready**: Model saved with `joblib` for reusability  

---

## 📂 Project Structure
├── sales_data/
│ ├── sales_data.csv
│ ├── calendar_data_full.csv
│ └── best_grocery_model.pkl
├── grocery_predictor.ipynb # Main Notebook with dashboard
├── grocery_model.py # Python script version
└── README.m