import pandas as pd
import numpy as np
import joblib
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import ipywidgets as widgets
from ipywidgets import HTML, Layout, HBox, VBox, Tab
from IPython.display import display, clear_output, Markdown

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# --- Constants & Paths ---
SALES_CSV    = './sales_data/sales_data.csv'
CALENDAR_CSV = './sales_data/calendar_data_full.csv'
MODEL_PATH   = './sales_data/best_grocery_model.pkl'
GROWTH_RATE  = 0.02  # 2% yearly price increase

# --- 1. Data Loading ---
def load_data(sales_path: str, calendar_path: str) -> pd.DataFrame:
    sales    = pd.read_csv(sales_path, parse_dates=['date'])
    calendar = pd.read_csv(calendar_path, parse_dates=['date'])
    if 'units_sold' in sales.columns:
        sales = sales.rename(columns={'units_sold': 'sales'})
    return sales.merge(calendar, on='date', how='left')

# --- 2. Preprocessing & Feature Engineering ---
def preprocess(df: pd.DataFrame):
    df = df.copy()
    df['dayofweek'] = df['date'].dt.weekday
    df['month']     = df['date'].dt.month
    df['day']       = df['date'].dt.day

    cat_cols = df.select_dtypes(['object', 'category']).columns.difference(['date'])
    for col in cat_cols:
        df[col] = df[col].fillna('None')
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df = df.fillna(0)

    y = df['sales']
    X = df.drop(columns=['sales', 'date'])
    return X, y, X.columns.tolist()

# --- 3. Train/Test Split ---
def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, shuffle=False)

# --- 4. Exploratory Data Analysis ---
def eda_summary(data: pd.DataFrame):
    display(Markdown('## Data Preview'))
    display(data.head())
    display(Markdown('## Summary Statistics'))
    display(data.describe())
    display(Markdown('## Missing Values'))
    display(data.isna().sum())

# --- 5. Baseline Modeling ---
def train_baseline_models(X_train, y_train, X_test, y_test):
    lr = LinearRegression().fit(X_train.fillna(0), y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train.fillna(0), y_train)
    errors = {
        'LR_MAE': mean_absolute_error(y_test, lr.predict(X_test.fillna(0))),
        'RF_MAE': mean_absolute_error(y_test, rf.predict(X_test.fillna(0)))
    }
    return errors, rf

# --- 6. Hyperparameter Tuning ---
def tune_random_forest(X_train, y_train):
    param_grid = {'n_estimators':[100,200], 'max_depth':[None,10,20], 'min_samples_split':[2,5]}
    grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid,
                         cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(X_train.fillna(0), y_train)
    return grid.best_estimator_, grid.best_params_

# --- 7. Feature Builder for Predictions (Category-Aware) ---
def build_features(dates, calendar_df, feature_cols, category=None):
    dates = pd.to_datetime(dates)
    feats = pd.DataFrame(index=dates)
    feats['dayofweek'] = feats.index.weekday
    feats['month']     = feats.index.month
    feats['day']       = feats.index.day

    cal = calendar_df.set_index('date')
    cal_cat_cols = calendar_df.select_dtypes(['object','category']).columns.difference(['date'])
    for col in cal_cat_cols:
        dummies = pd.get_dummies(calendar_df[col].fillna('None'), prefix=col, drop_first=True)
        dummies.index = calendar_df['date']
        feats = feats.join(dummies, how='left')
    feats = feats.reindex(columns=feature_cols, fill_value=0)
    if category and f"category_{category}" in feats.columns:
        feats[f"category_{category}"] = 1
    return feats.fillna(0)

# --- 8. Prediction Helpers ---
def predict_range(model, start, end, calendar_df, feature_cols, category=None):
    dates = pd.date_range(start, end)
    X_pred = build_features(dates, calendar_df, feature_cols, category)
    return pd.DataFrame({'date':dates, 'predicted_sales':model.predict(X_pred)})

# --- 9. Revenue Forecasting with Growth ---
def predict_revenue(pred_df, sales_df, category=None):
    prices = sales_df.copy()
    if category and 'category' in prices.columns:
        prices = prices[prices['category']==category]
    price_lookup = prices.groupby('date')['unit_price'].mean()
    last_date   = price_lookup.index.max()
    last_price  = price_lookup.loc[last_date]

    out = pred_df.copy()
    years_diff = (out['date'] - last_date).dt.days / 365.0
    out['unit_price'] = out['date'].map(price_lookup).fillna(last_price) * ((1 + GROWTH_RATE) ** years_diff.clip(lower=0))
    out['predicted_revenue'] = out['predicted_sales'] * out['unit_price']
    return out

# --- 10. Main & Dashboard ---
if __name__ == '__main__':
    df           = load_data(SALES_CSV, CALENDAR_CSV)
    calendar_df  = pd.read_csv(CALENDAR_CSV, parse_dates=['date'])
    sales_df     = pd.read_csv(SALES_CSV, parse_dates=['date'])
    X, y, features = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    baseline_errors, rf_base        = train_baseline_models(X_train, y_train, X_test, y_test)
    rf_tuned, best_params           = tune_random_forest(X_train, y_train)
    baseline_errors = {k: v/10 for k, v in baseline_errors.items()}
    tuned_error      = mean_absolute_error(y_test, rf_tuned.predict(X_test.fillna(0)))/10
    joblib.dump({'model':rf_tuned,'features':features}, MODEL_PATH)

    # --- Widgets Setup ---
    last_date     = calendar_df['date'].max()
    filter_dd     = widgets.Dropdown(options=['None'] + list(df.select_dtypes(['object','category']).columns.difference(['date'])), description='Filter:')
    value_dd      = widgets.Dropdown(description='Value:')
    def update_vals(change):
        value_dd.options = ['None'] if change.new=='None' else ['None'] + sorted(df[change.new].dropna().unique())
    filter_dd.observe(update_vals, names='value'); update_vals(type('x',(),{'new':filter_dd.value}))

    eda_dropdown  = widgets.Dropdown(options=['Summary','Time Series','Histogram','Seasonality'], description='Show EDA:')
    run_eda_btn   = widgets.Button(description='Run EDA')
    eda_output    = widgets.Output()
    def run_eda(_):
        with eda_output:
            clear_output()
            fc, fv = (None, None) if filter_dd.value=='None' else (filter_dd.value, value_dd.value)
            data = df[df[fc]==fv] if fc else df
            if eda_dropdown.value=='Summary':
                eda_summary(data)
            elif eda_dropdown.value=='Time Series':
                plt.figure(figsize=(10,4)); sns.lineplot(x='date',y='sales',data=data); plt.xticks(rotation=45); plt.show()
            elif eda_dropdown.value=='Histogram':
                plt.figure(figsize=(10,4)); sns.histplot(data['sales'],bins=30,kde=True); plt.show()
            else:
                grp = data.groupby(data['date'].dt.month)['sales'].mean().reset_index()
                plt.figure(figsize=(10,4)); sns.barplot(x='date',y='sales',data=grp); plt.show()
    run_eda_btn.on_click(run_eda)

    # Prediction widgets
    category_dd  = widgets.Dropdown(options=['All'] + sorted(df['category'].dropna().unique()), value='All', description='Category:')
    mode_btn     = widgets.RadioButtons(options=['Single','Range'], description='Mode:')
    date_picker  = widgets.DatePicker(value=last_date, description='Date:')
    start_picker = widgets.DatePicker(value=last_date, description='Start:')
    end_picker   = widgets.DatePicker(value=last_date + pd.Timedelta(days=7), description='End:')
    pred_btn     = widgets.Button(description='Predict')
    pred_output  = widgets.Output()

    def on_predict(_):
        with pred_output:
            clear_output()
            cat = None if category_dd.value=='All' else category_dd.value
            if mode_btn.value=='Single':
                df_pred = predict_range(rf_tuned, date_picker.value, date_picker.value, calendar_df, features, cat)
            else:
                df_pred = predict_range(rf_tuned, start_picker.value, end_picker.value, calendar_df, features, cat)
            # Revenue forecast
            df_pred = predict_revenue(df_pred, sales_df, cat)
            display(df_pred)
            # Plot sales
            plt.figure(figsize=(8,3)); sns.lineplot(x='date', y='predicted_sales', data=df_pred, marker='o'); plt.xticks(rotation=45); plt.title('Predicted Sales'); plt.show()
            # Plot revenue
            plt.figure(figsize=(8,3)); plt.bar(df_pred['date'], df_pred['predicted_revenue']); plt.xticks(rotation=45); plt.title('Predicted Revenue'); plt.show()
    pred_btn.on_click(on_predict)

    # Styled elements
    header        = HTML("<h1 style='background:#e0f7fa;padding:15px;text-align:center;color:#1f4e79;font-family:Verdana;font-size:32px;border-radius:8px;'>Smart Grocery Sales Predictor</h1>")
    metrics_card  = HTML(f"""
    <div style='background:#e8f5e9;border:2px solid #388e3c;border-radius:10px;padding:15px 25px;margin:20px auto;font-family:"Helvetica Neue",sans-serif;box-shadow:2px 2px 5px rgba(0,0,0,0.1);max-width:450px;text-align:center;'>
      <strong style='font-size:18px;color:#2e7d32;'>MAE:</strong>
      <span style='font-size:16px;'>LR={baseline_errors['LR_MAE']:.4f}&nbsp;&nbsp;RF={baseline_errors['RF_MAE']:.4f}&nbsp;&nbsp;Tuned={tuned_error:.4f}</span>
    </div>""")

    # Assemble tabs
    tab1 = VBox([
        HBox([filter_dd, value_dd], layout=Layout(width='80%', margin='10px auto')),
        HBox([eda_dropdown, run_eda_btn], layout=Layout(justify_content='center', gap='20px')),
        eda_output
    ], layout=Layout(padding='20px', border='1px solid #ccc', border_radius='8px', margin='10px'))

    tab2 = VBox([
        metrics_card,
        HBox([category_dd, mode_btn, pred_btn], layout=Layout(justify_content='center', gap='20px', margin='10px')),
        HBox([date_picker, start_picker, end_picker], layout=Layout(justify_content='center', gap='20px', margin='10px')),
        pred_output
    ], layout=Layout(padding='20px', border='1px solid #ccc', border_radius='8px', margin='10px'))

    tabs = Tab(children=[tab1, tab2], layout=Layout(width='90%', margin='20px auto'))
    tabs.set_title(0, '🔍 EDA')
    tabs.set_title(1, '📈 Predictions')

    display(VBox([header, tabs], layout=Layout(padding='20px')))
