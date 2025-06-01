import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
from sklearn.model_selection import train_test_split
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# --- 1. Database Connection ---
db_params = {
    'host': 'localhost',
    'database': 'GP_DB',
    'user': 'ishani',
    'password': 'read123'
}

conn = None
try:
    conn = psycopg2.connect(**db_params)
    print("Successfully connected to PostgreSQL!")
except psycopg2.Error as e:
    print(f"Error connecting or executing SQL: {e}")

# --- 2. Function to Fetch Data ---
def fetch_platform_data(table_name, platform_name):
    try:
        conn = psycopg2.connect(**db_params)
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql(query, conn)
        df['platform'] = platform_name
        return df
    except psycopg2.Error as e:
        print(f"Error fetching data from {table_name}: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

# --- 3. Fetching Game and Price Data ---
ps_df = fetch_platform_data('games_ps', 'PlayStation')
steam_df = fetch_platform_data('games_st', 'Steam')
xbox_df = fetch_platform_data('games_xb', 'Xbox')

ps_price_df = fetch_platform_data('prices_ps', 'PlayStation')
steam_price_df = fetch_platform_data('prices_st', 'Steam')
xbox_price_df = fetch_platform_data('prices_xb', 'Xbox')

# --- 4. Merging Data ---
all_games_df = pd.concat([ps_df, steam_df, xbox_df], ignore_index=True)
all_prices_df = pd.concat([ps_price_df, steam_price_df, xbox_price_df], ignore_index=True)
merged_df = pd.merge(all_games_df, all_prices_df, on=['gameid', 'platform'], how='inner')
print("Merged data for all platforms.")

# --- 5. Cleaning Columns ---
merged_df['release_date'] = pd.to_datetime(merged_df['release_date'], errors='coerce')
merged_df = merged_df.dropna(subset=['release_date', 'usd'])

# --- 5.1 Filtering for Top 20 Publishers ---
top_publishers = merged_df['publishers'].value_counts().head(20).index.tolist()
merged_df = merged_df[merged_df['publishers'].isin(top_publishers)]
print("Top 20 Punlishers found. ")

# --- 6. Removing Price Outliers ---
merged_df = merged_df[merged_df['usd'] <= 100]
print("Outliers Removed")

# --- 7. Sort and Sample (15%) ---
merged_df = merged_df.sort_values('release_date')
merged_df = merged_df.sample(frac=0.10, random_state=42).reset_index(drop=True)
print("Sorting and sampling completed.")

# --- 8. Feature Engineering ---
merged_df['days_since_release'] = (datetime.today() - merged_df['release_date']).dt.days
df_model = pd.get_dummies(merged_df, columns=['platform', 'publishers'], drop_first=False)
print("Feature Engineering Completed.")

# --- 9. Train/Test Split ---
platform_cols = [col for col in df_model.columns if col.startswith('platform_')]
publisher_cols = [col for col in df_model.columns if col.startswith('publishers_')]
X = df_model[['days_since_release'] + platform_cols+ publisher_cols]
y = df_model['usd']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train Test Split Completed.")

# --- 10. Random Forest ---
rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest Model Completed.")

# --- 11. XGBoost ---
xgb = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, verbosity=0)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
print("XGBoost Model Completed.")

# --- 12. Prophet ---
df_prophet = merged_df[['release_date', 'usd']].rename(columns={'release_date': 'ds', 'usd': 'y'})
prophet_model = Prophet()
prophet_model.fit(df_prophet)
future = prophet_model.make_future_dataframe(periods=30)
forecast = prophet_model.predict(future)
merged_prophet = pd.merge(df_prophet, forecast[['ds', 'yhat']], on='ds', how='inner')
prophet_mae = mean_absolute_error(merged_prophet['y'], merged_prophet['yhat'])
prophet_rmse = mean_squared_error(merged_prophet['y'], merged_prophet['yhat'], squared=False)
prophet_r2 = r2_score(merged_prophet['y'], merged_prophet['yhat'])
print("Prophet Model Completed")

# --- 13. ARIMA ---
arima_series = merged_df.set_index('release_date')['usd']
model_arima = ARIMA(arima_series, order=(3, 1, 2)).fit()
forecast_arima = model_arima.forecast(steps=30)
true_vals = arima_series[-30:] if len(arima_series) > 30 else arima_series
arima_mae = mean_absolute_error(true_vals, forecast_arima[:len(true_vals)])
arima_rmse = mean_squared_error(true_vals, forecast_arima[:len(true_vals)], squared=False)
arima_r2 = r2_score(true_vals, forecast_arima[:len(true_vals)])
print("ARIMA Model Completed")

# --- 14. Model Evaluation ---
results = pd.DataFrame({
    'Model': ['ARIMA', 'Random Forest', 'XGBoost', 'Prophet'],
    'MAE': [arima_mae, mean_absolute_error(y_test, rf_pred), mean_absolute_error(y_test, xgb_pred), prophet_mae],
    'RMSE': [arima_rmse, mean_squared_error(y_test, rf_pred, squared=False), mean_squared_error(y_test, xgb_pred, squared=False), prophet_rmse],
    'R2': [arima_r2, r2_score(y_test, rf_pred), r2_score(y_test, xgb_pred), prophet_r2]
})

print("\nModel Comparison:")
print(results.round(3))
