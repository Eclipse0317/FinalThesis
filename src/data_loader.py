import pandas as pd

def load_data(cny_path, cnh_path):
    """加载并清洗investing.com的周度数据"""
    cny = pd.read_csv(cny_path)
    cnh = pd.read_csv(cnh_path)

    for df in [cny, cnh]:
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        df.set_index("Date", inplace=True)
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    weekly = cny[["Price"]].rename(columns={"Price": "CNY"}).join(
        cnh[["Price"]].rename(columns={"Price": "CNH"}), how="inner"
    )

    print(f"数据范围: {weekly.index[0].date()} 到 {weekly.index[-1].date()}")
    print(f"观测数: {len(weekly)}")
    print(f"缺失值: CNY={weekly['CNY'].isna().sum()}, CNH={weekly['CNH'].isna().sum()}")

    return weekly