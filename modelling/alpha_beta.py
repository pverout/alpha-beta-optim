import yfinance as yf
import statsmodels.api as sm
import pandas as pd

class AlphaBeta:
    def __init__(self, benchmark_ticker="^GSPC"):
        self.benchmark_ticker = benchmark_ticker

    def _get_benchmark_returns(self, start, end):
        # Download adjusted close prices for the benchmark within the date range
        benchmark_data = yf.download(self.benchmark_ticker, start=start, end=end)["Adj Close"]
        benchmark_returns = benchmark_data.pct_change().dropna()
        return benchmark_returns

    def calculate_alpha_beta(self, portfolio):
        # Get benchmark returns for the same date range as the portfolio
        benchmark_returns = self._get_benchmark_returns(portfolio.start, portfolio.end)

        # Extract portfolio returns and align dates with benchmark returns
        portfolio_returns = portfolio.returns.mean(axis=1)
        
        # Align both dataframes by date to ensure same size
        data = pd.DataFrame({
            "Portfolio": portfolio_returns,
            "Benchmark": benchmark_returns
        }).dropna()

        # Add a constant for the intercept in OLS regression
        X = sm.add_constant(data["Benchmark"])
        y = data["Portfolio"]

        # Fit the OLS model
        model = sm.OLS(y, X).fit()
        alpha, beta = model.params['const'], model.params['Benchmark']
        
        return alpha, beta

# Example Usage
portfolio = Portfolio(
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN"],
    weights=[0.3, 0.2, 0.3, 0.2],
    start="2013-12-31",
    end="2021-01-01"
)

alphabeta = AlphaBeta()
alpha, beta = alphabeta.calculate_alpha_beta(portfolio)

print(f"Portfolio Alpha: {alpha:.4f}")
print(f"Portfolio Beta: {beta:.4f}")