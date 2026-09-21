import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def load_or_generate_sales_data(filepath: str = "data/sales_data.csv", days: int = 30) -> pd.DataFrame:
    """
    Loads POS sales data from CSV. If the file is not found, generates
    a realistic 30-day daily sales history for soft drink SKUs.
    """
    try:
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except FileNotFoundError:
        # Fallback generator simulating store sales (e.g., Kaggle Corporación Favorita style)
        np.random.seed(42)
        end_date = datetime.now().date()
        date_range = [end_date - timedelta(days=i) for i in range(days)][::-1]
        
        products = ["Coca-Cola", "Diet Coke", "Sprite", "Fanta"]
        base_demand = {"Coca-Cola": 55, "Diet Coke": 35, "Sprite": 30, "Fanta": 20}
        
        records = []
        for d in date_range:
            day_of_week = d.weekday()
            weekend_boost = 1.3 if day_of_week in [4, 5, 6] else 1.0  # Fri-Sun surge
            
            for prod in products:
                noise = np.random.normal(0, 3)
                potential_demand = int(max(5, (base_demand[prod] * weekend_boost) + noise))
                records.append({
                    "date": pd.to_datetime(d),
                    "store_id": 1,
                    "brand_name": prod,
                    "baseline_demand": potential_demand,
                    "actual_sales": potential_demand,
                    "compliance_score": 100.0,
                    "has_stockout": False,
                    "facing_gap": 0
                })
        return pd.DataFrame(records)

def fuse_shelf_audit_with_sales(
    sales_df: pd.DataFrame, 
    comparison_df: pd.DataFrame, 
    overall_compliance: float,
    audit_date: datetime.date = None
) -> pd.DataFrame:
    """
    Fuses the VLM audit results into the sales time-series for the audit date.
    Simulates demand censorship: when a stockout occurs, actual sales drop.
    """
    df = sales_df.copy()
    if audit_date is None:
        audit_date = df["date"].max()
    else:
        audit_date = pd.to_datetime(audit_date)

    audit_mask = df["date"] == audit_date

    for _, row in comparison_df.iterrows():
        brand = row["brand_name"]
        gap = row["facing_gap"]
        stockout = row["has_stockout"]
        item_score = row["compliance_score"]
        
        mask = audit_mask & (df["brand_name"].str.lower() == brand.lower())
        
        if mask.any():
            df.loc[mask, "compliance_score"] = item_score
            df.loc[mask, "has_stockout"] = stockout
            df.loc[mask, "facing_gap"] = gap
            
            # Simulate Demand Censorship:
            # If stockout occurs, sales dip by 60-80% of true potential demand
            if stockout or gap > 0:
                lost_ratio = 0.75 if stockout else min(0.4, gap * 0.15)
                base = df.loc[mask, "baseline_demand"].values[0]
                censored_sales = int(round(base * (1.0 - lost_ratio)))
                df.loc[mask, "actual_sales"] = censored_sales

    # Calculate lost revenue column (assuming an average retail price of $2.50 per unit)
    df["lost_units"] = df["baseline_demand"] - df["actual_sales"]
    df["lost_revenue"] = df["lost_units"] * 2.50
    return df