import yfinance as yf
import statsmodels.api as sm
import pandas as pd
from data.portfolio import Portfolio
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.portfolio import Portfolio


class AlphaBeta:
    def __init__(self, benchmark_ticker="^GSPC", risk_free_rate=0.0138):
        self.benchmark_ticker = benchmark_ticker
        self.risk_free_rate = risk_free_rate

    def _get_benchmark_returns(self, start, end):
        # Download adjusted close prices for the benchmark within the date range
        benchmark_data = yf.download(self.benchmark_ticker, start=start, end=end)["Adj Close"]
        benchmark_returns = benchmark_data.pct_change().dropna()
        return benchmark_returns

    def calculate_CAPM(self, portfolio):
        # Get benchmark returns for the same date range as the portfolio
        benchmark_returns = self._get_benchmark_returns(portfolio.start, portfolio.end)

        # Calculate weighted portfolio returns
        portfolio_returns = (portfolio.returns * portfolio.weights).sum(axis=1)
        
        portfolio_returns = portfolio_returns.squeeze()  # Convert to 1D if it's a 2D array with a single column
        benchmark_returns = benchmark_returns.squeeze()

        # Align both dataframes by date to ensure same size
        data = pd.DataFrame({
            "Portfolio": portfolio_returns,
            "Benchmark": benchmark_returns
        }).dropna()

        # Calculate excess returns
        data["Excess Portfolio"] = data["Portfolio"] - self.risk_free_rate
        data["Excess Benchmark"] = data["Benchmark"] - self.risk_free_rate
        # Perform OLS regression with excess benchmark as the independent variable
        X = sm.add_constant(data["Excess Benchmark"])  # Adding a constant for the intercept (alpha)
        y = data["Excess Portfolio"]

        # Fit the OLS model to find alpha and beta
        model = sm.OLS(y, X).fit()
        alpha, beta = model.params['const'], model.params['Excess Benchmark']
        
        return alpha, beta



        # def calculate_capm(stock_returns, market_returns, risk_free_rate):
        # """
        # Calculate the expected return of a stock using the CAPM model.

        # Parameters:
        # - stock_returns: A pandas Series of the stock's returns
        # - market_returns: A pandas Series of the market's returns
        # - risk_free_rate: The risk-free rate (float)

        # Returns:
        # - expected_return: The expected return of the stock according to CAPM
        # - beta: The calculated beta of the stock
        # """

        # # Ensure inputs are in the right format
        # stock_returns = pd.Series(stock_returns)
        # market_returns = pd.Series(market_returns)

        # # Add a constant for the OLS model (intercept)
        # X = sm.add_constant(market_returns)

        # # Fit the OLS model
        # model = sm.OLS(stock_returns, X).fit()

        # # Extract beta (slope coefficient) and intercept (alpha)
        # beta = model.params[1]  # the slope of the regression line
        # alpha = model.params[0]  # the intercept of the regression line

        # # Calculate expected return using CAPM formula
        # expected_return = risk_free_rate + beta * (market_returns.mean() - risk_free_rate)

        # return expected_return, beta

# Example Usage
portfolio = Portfolio(
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN"],
    weights=[0.3, 0.2, 0.3, 0.2],
    start="2013-12-31",
    end="2021-01-01"
)


alphabeta = AlphaBeta()
alpha, beta = alphabeta.calculate_CAPM(portfolio)

print(f"Portfolio Alpha: {alpha:.4f}")
print(f"Portfolio Beta: {beta:.4f}")