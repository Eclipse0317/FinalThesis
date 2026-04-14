import yfinance as yf

# Define the tickers for CNH and CNY
tickers = ["CNH=X", "CNY=X"]

# Fetch weekly data
weekly_data = yf.download(tickers, start="2015-01-01", end="2026-02-28", interval="1wk")

# Fetch monthly data
# monthly_data = yf.download(tickers, start="2015-01-01", end="2026-02-28", interval="1mo")

# --- New code to save to CSV ---

# Save the weekly data
weekly_data.to_csv("cnh_cny_weekly_data.csv")

# Save the monthly data
# monthly_data.to_csv("cnh_cny_monthly_data.csv")

print("Files successfully saved!")