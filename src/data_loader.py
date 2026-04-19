import pandas as pd
import numpy as np
from src.config import CNY_DATA_FILE, CNH_DATA_FILE

def load_data():
    """
    Loads weekly USD/CNY and USD/CNH data, aligns them, 
    and calculates log returns.
    """
    # 1. Load raw CSVs
    cny = pd.read_csv(CNY_DATA_FILE)
    cnh = pd.read_csv(CNH_DATA_FILE)

    # 2. Clean dates and indices
    for df in [cny, cnh]:
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        df.set_index("Date", inplace=True)
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # 3. Merge into a single DataFrame
    weekly = cny[["Price"]].rename(columns={"Price": "CNY"}).join(
        cnh[["Price"]].rename(columns={"Price": "CNH"}), how="inner"
    )

    # 4. Feature Engineering: Calculate Log Returns (scaled by 100)
    weekly["r_CNY"] = np.log(weekly["CNY"]).diff() * 100
    weekly["r_CNH"] = np.log(weekly["CNH"]).diff() * 100

    # 5. Drop missing values
    weekly.dropna(inplace=True)
    weekly = weekly.asfreq('W-SUN').ffill()

    print(f"Data Loaded: {len(weekly)} observations from {weekly.index[0].date()} to {weekly.index[-1].date()}")
    
    return weekly