import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def load_or_generate_sales_data(filepath: str = "clean_retail_sales.csv", days: int = 30) -> pd.DataFrame:
    """
    Loads POS sales data from the wrangled Kaggle CSV. 
    Dates and volumes are already perfectly scaled in the CSV.
    Dynamically shifts the dates so the dataset always ends on 'today'.
    Injects historical compliance variances to power the UI heatmaps and trendlines.
    """
    try:
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["date"])
        
        # --- DYNAMIC DATE SHIFT ---
        # Calculates the gap between the CSV's max date and today, and shifts all rows forward
        date_delta = pd.Timestamp.today().normalize() - df["date"].max()
        df["date"] = df["date"] + date_delta
        
        # Re-sync the weekday labels so the Weekend/Weekday chart in Tab 3 stays accurate
        df["weekday"] = df["date"].dt.day_name()
        df["is_weekend"] = df["date"].dt.dayofweek >= 5

        # We must generate these historical metrics so Tab 2 (Heatmap/Trends) doesn't crash
        if "has_stockout" not in df.columns:
            np.random.seed(42)
            # If Claude's data shows actual sales dropped below baseline, it was a stockout
            df["has_stockout"] = df["actual_sales"] < df["baseline_demand"]
            df["facing_gap"] = df["has_stockout"].apply(lambda x: np.random.randint(1, 4) if x else 0)
            df["compliance_score"] = df["has_stockout"].apply(
                lambda x: np.random.uniform(50, 85) if x else np.random.uniform(90, 100)
            ).round(1)
            
        return df
        
    except FileNotFoundError:
        # Fallback generator simulating store sales for all aisles
        np.random.seed(42)
        end_date = datetime.now().date()
        date_range = [end_date - timedelta(days=i) for i in range(days)][::-1]
        
        products = ["Coca-Cola", "Diet Coke", "Sprite", "Fanta", "Doritos", "Lays", "Colgate Toothpaste", "Dove Soap"]
        base_demand = {
            "Coca-Cola": 12, "Diet Coke": 8, "Sprite": 10, "Fanta": 6,
            "Doritos": 14, "Lays": 12,
            "Colgate Toothpaste": 4, "Dove Soap": 5
        }
        
        records = []
        for d in date_range:
            day_of_week = d.weekday()
            weekend_boost = 1.4 if day_of_week in [4, 5, 6] else 1.0  # Fri-Sun surge
            
            for prod in products:
                noise = np.random.normal(0, 2)
                potential_demand = int(max(2, (base_demand[prod] * weekend_boost) + noise))
                
                # Introduce historical stockouts to populate the heatmap
                is_stockout = np.random.random() < 0.12 # 12% chance of historical stockout
                comp_score = np.random.uniform(50, 85) if is_stockout else np.random.uniform(90, 100)
                actual = int(potential_demand * 0.4) if is_stockout else potential_demand
                gap = np.random.randint(1, 4) if is_stockout else 0

                records.append({
                    "date": pd.to_datetime(d),
                    "store_id": "CA_1",
                    "brand_name": prod,
                    "baseline_demand": potential_demand,
                    "actual_sales": actual,
                    "compliance_score": round(comp_score, 1),
                    "has_stockout": is_stockout,
                    "facing_gap": gap
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

    # Dynamic pricing for the 3 distinct aisles
    prices = {
        "Coca-Cola": 2.50, "Diet Coke": 2.50, "Sprite": 2.50, "Fanta": 2.50,
        "Doritos": 4.50, "Lays": 4.50,
        "Colgate Toothpaste": 6.50, "Dove Soap": 5.00
    }

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

    # Calculate lost revenue using the exact item prices
    df["price"] = df["brand_name"].map(prices).fillna(3.00)
    df["lost_units"] = df["baseline_demand"] - df["actual_sales"]
    df["lost_revenue"] = df["lost_units"] * df["price"]
    
    return df