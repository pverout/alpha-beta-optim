#%%
import yfinance as yf
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]  # Add any other tickers you want here

# Download data for multiple stocks
data = yf.download(tickers, start="2013-12-31", end="2021-01-01")

# Display the first few rows of the data
print(data.head())
# %%
