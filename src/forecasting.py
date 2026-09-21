import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import timedelta

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts ML features from dates (day of week, weekend)."""
    df_feat = df.copy()
    df_feat["day_of_week"] = df_feat["date"].dt.dayofweek
    df_feat["is_weekend"] = df_feat["day_of_week"].apply(lambda x: 1 if x >= 4 else 0) # Fri, Sat, Sun
    return df_feat

def train_and_forecast(df: pd.DataFrame, brand: str, forecast_days: int = 7) -> pd.DataFrame:
    """
    Trains an XGBoost model on historical data to predict true baseline demand, 
    then generates a 7-day forecast.
    """
    # 1. Filter historical data for the specific brand
    brand_df = df[df["brand_name"] == brand].copy()
    brand_df = prepare_features(brand_df)
    
    # 2. Define Features (X) and Target (y)
    # We predict 'baseline_demand' (true demand) so the store knows what to actually stock
    features = ["day_of_week", "is_weekend"]
    X = brand_df[features]
    y = brand_df["baseline_demand"]
    
    # 3. Train the XGBoost Model
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    
    # 4. Generate Future Dates for the next 7 days
    last_date = brand_df["date"].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
    
    future_df = pd.DataFrame({"date": future_dates})
    future_df = prepare_features(future_df)
    
    # 5. Make Predictions
    future_predictions = model.predict(future_df[features])
    
    # Clean up predictions (demand can't be negative, and round to nearest whole item)
    future_df["forecasted_demand"] = np.maximum(0, np.round(future_predictions))
    future_df["brand_name"] = brand
    
    return future_df[["date", "brand_name", "forecasted_demand"]]