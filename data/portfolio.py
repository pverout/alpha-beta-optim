import yfinance as yf
import pandas as pd
import numpy as np

class Portfolio:
    def __init__(self, tickers, weights, start, end):
        # Ensure weights sum to 1
        if sum(weights) != 1:
            raise ValueError("Weights must sum to 1.")
        self.tickers = tickers
        self.weights = weights
        self.start = start
        self.end = end
        self.data = self._download_data()
        self.returns = self._calculate_returns()

    def _download_data(self):
        # Download adjusted close prices for the given tickers and date range
        data = yf.download(self.tickers, start=self.start, end=self.end)["Adj Close"]
        return data

    def _calculate_returns(self):
        # Calculate daily returns for each stock
        returns = self.data.pct_change().dropna()
        return returns

    def portfolio_return(self):
        # Calculate the weighted portfolio return
        weighted_returns = (self.returns * self.weights).sum(axis=1)
        portfolio_return = weighted_returns.mean() * 252  # Annualized return
        return portfolio_return

    def portfolio_volatility(self):
        # Calculate the portfolio volatility
        weighted_cov_matrix = np.dot(self.weights, np.dot(self.returns.cov() * 252, self.weights))
        portfolio_volatility = np.sqrt(weighted_cov_matrix)
        return portfolio_volatility

    def portfolio_performance(self):
        # Returns the annualized portfolio return and volatility
        return self.portfolio_return(), self.portfolio_volatility()

if __name__ == "__main__":
    # Example Usage
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    weights = [0.3, 0.2, 0.3, 0.2]  # Adjust these as long as they sum to 1
    start_date = "2013-12-31"
    end_date = "2021-01-01"

    portfolio = Portfolio(tickers, weights, start_date, end_date)
    annual_return, annual_volatility = portfolio.portfolio_performance()

    print(f"Annual Portfolio Return: {annual_return:.2%}")
    print(f"Annual Portfolio Volatility: {annual_volatility:.2%}")