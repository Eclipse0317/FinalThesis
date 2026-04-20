import pandas as pd
import numpy as np
from src.config import CNY_DATA_FILE, CNH_DATA_FILE

def load_data():
    """
    Loads weekly USD/CNY and USD/CNH data, aligns them, 
    and calculates log returns.
    
    Removes CNY "frozen" weeks (all OHLC identical) which correspond to
    Chinese holiday weeks when the onshore market is closed. On these
    weeks, CNY return = 0 is an artifact of market closure, not a real
    price signal, while CNH continues to trade offshore. Leaving them in
    would create spurious basis risk.
    """
    # 1. Load raw CSVs (keep OHLC for frozen-week detection)
    cny = pd.read_csv(CNY_DATA_FILE)
    cnh = pd.read_csv(CNH_DATA_FILE)

    # 2. Clean dates and prices
    for df in [cny, cnh]:
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        df.set_index("Date", inplace=True)
        for col in ["Price", "Open", "High", "Low"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Flag and drop CNY frozen-quote weeks (holiday weeks with no onshore trading)
    cny_frozen = (
        (cny["Open"] == cny["High"]) 
        & (cny["High"] == cny["Low"]) 
        & (cny["Low"] == cny["Price"])
    )
    if cny_frozen.any():
        print(f"Dropping {cny_frozen.sum()} CNY frozen-quote weeks "
              f"(holiday weeks): {cny.index[cny_frozen].strftime('%Y-%m-%d').tolist()}")
        cny = cny.loc[~cny_frozen]

    # 4. Merge (now on cleaned CNY)
    weekly = cny[["Price"]].rename(columns={"Price": "CNY"}).join(
        cnh[["Price"]].rename(columns={"Price": "CNH"}), how="inner"
    )

    # 4. Feature Engineering: Calculate Log Returns (scaled by 100)
    weekly["r_CNY"] = np.log(weekly["CNY"]).diff() * 100
    weekly["r_CNH"] = np.log(weekly["CNH"]).diff() * 100

    # 5. Drop missing values
    weekly.dropna(inplace=True)

    print(f"Data Loaded: {len(weekly)} observations from {weekly.index[0].date()} to {weekly.index[-1].date()}")
    
    return weekly