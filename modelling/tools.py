import sys
import os
import numpy as np
import yfinance as yf
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.portfolio import Portfolio
from scipy.optimize import minimize
import statsmodels.api as sm

class Tools:
    def __init__(self, portfolio):
        self.portfolio = portfolio



    def simulate_random_portfolios(self, num_portfolios, alpha = 0.05):
        results = np.zeros((5, num_portfolios))  # Store returns, std dev, and sharpe ratio
        weights_array = np.zeros((len(self.portfolio.tickers), num_portfolios))  # Store weights for each portfolio

        for i in range(num_portfolios):
            weights = np.random.random(len(self.portfolio.tickers))
            weights = weights/np.sum(weights)  # Normalize to sum to 1
            portfolio_return, portfolio_std, sharpe_ratio, vaR, cvaR = self.portfolio.portfolio_performance(alpha = alpha, weights = weights)
            results[0, i] = portfolio_return
            results[1, i] = portfolio_std
            results[2, i] = sharpe_ratio
            results[3, i] = vaR
            results[4, i] = cvaR
            weights_array[:, i] = weights
        return results, weights_array


    def optimize_metric(self, metric='sharpe', alpha=0.05):

        def objective(weights):
            if metric == 'sharpe':
                return -self.portfolio.portfolio_shape(weights = weights)  # Negative for maximization
            elif metric == 'var':
                return self.portfolio.portfolio_VaR(alpha = alpha, weights = weights)
            elif metric == 'cvar':
                return self.portfolio.portfolio_CVaR(alpha = alpha, weights = weights)
            elif metric == 'shape_var':
                return -self.portfolio.portfolio_shape_VaR(alpha = alpha, weights = weights)
            elif metric == 'shape_cvar':
                return -self.portfolio.portfolio_shape_CVaR(alpha = alpha, weights = weights)
            # else:
            #     raise ValueError("Metric not recognized. Choose from 'sharpe', 'var', or 'cvar'.")

        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights must sum to 1
        bounds = tuple((0, 1) for _ in range(len(self.portfolio.tickers)))  # Weights between 0 and 1

        initial_guess = self.portfolio.weights
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            return result.x, -result.fun  # Return optimized weights and the metric value (negated if it's Sharpe)
        else:
            raise ValueError("Optimization failed.")

    def calculate_alpha_beta(self, ticker = "^GSPC", weights=None):
        # If weights are not provided, use the portfolio's weights
        market_data = yf.download(ticker, start=self.portfolio.start, end=self.portfolio.end)["Adj Close"]
        market_returns = market_data.pct_change().dropna()
        if weights is None:
            weights = self.portfolio.weights

        # Calculate portfolio weighted returns
        portfolio_returns = (self.portfolio.returns * weights).sum(axis=1)

        # Calculate excess returns for the portfolio and the market
        excess_portfolio_returns = portfolio_returns - self.portfolio.risk_free_rate / 252
        excess_market_returns = market_returns - self.portfolio.risk_free_rate / 252

        # Perform OLS regression for CAPM
        X = sm.add_constant(excess_market_returns)
        y = excess_portfolio_returns
        model = sm.OLS(y, X).fit()

        # Extract alpha and beta from the model
        alpha = model.params.iloc[0]
        beta = model.params.iloc[1]

        return alpha, beta

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JNJ", "PFE", "VZ", "T", "KO"]
    weights = [1/len(tickers)] * len(tickers)  # Adjust these as long as they sum to 1
    start_date = "2013-12-31"
    end_date = "2021-01-01"
    risk_free_rate = 0.01
    portfolio = Portfolio(tickers, weights, start_date, end_date, risk_free_rate)

    tools = Tools(portfolio)
    num_portfolios = 10
    results, weights_array = tools.simulate_random_portfolios(num_portfolios)

    # Output some results
    print("Simulated Portfolio Returns:\n", results[0])
    print("Simulated Portfolio Standard Deviations:\n", results[1])
    print("Simulated Portfolio Sharpe Ratios:\n", results[2])
