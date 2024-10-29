import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm

class Portfolio:
    def __init__(self, data_in, tickers, weights, start, end,risk_free_rate):
        # Ensure weights sum to 1
        if np.round(sum(weights),2) != 1:
            raise ValueError("Weights must sum to 1.")
        self.tickers = tickers
        self.weights = weights
        self.start = start
        self.end = end
        self.risk_free_rate = risk_free_rate

        self.data = data_in['Adj Close'][self.tickers].loc[start:end]
        self.returns = self._calculate_return()
        self.stdev = self._calculate_stdev()
        self.mean_return = self.calculate_mean_returns()

    def _calculate_return(self):
        # Calculate daily returns for each stock
        returns = self.data.pct_change().dropna()
        return returns

    def calculate_mean_returns(self, weights=None):
        # Calculate the weighted portfolio return
        if weights is None:
            weights = self.weights  # Use the class variable if no weights are provided
        returns = (self.returns * weights).sum(axis=1)
        returns = returns.mean() * 252  # Annualized return
        return returns

    def _calculate_stdev(self, weights=None):
        # Calculate the portfolio standard deviation
        if weights is None:
            weights = self.weights  # Use the class variable if no weights are provided
        stdev = np.sqrt(np.dot(weights, np.dot(self.returns.cov() * 252, weights)))
        return stdev

    def portfolio_shape(self, weights=None):
        # Calculate the Sharpe ratio
        if weights is None:
            weights = self.weights  # Use the class variable if no weights are provided
        mean_return = self.calculate_mean_returns(weights)
        stdev = self._calculate_stdev(weights)
        shape = (mean_return - self.risk_free_rate) / stdev
        return shape

    def portfolio_VaR(self, alpha=0.05, weights=None):
        # Calculate the portfolio returns
        if weights is None:
            weights = self.weights  # Use the class variable if no weights are provided

        # Simulate portfolio returns using the covariance matrix
        port_returns = np.dot(self.returns, weights)
        # Calculate VaR
        var = np.abs(np.percentile(port_returns, 100 * alpha))
        return var

    def portfolio_CVaR(self, alpha=0.05, weights=None):
        # Calculate the portfolio returns
        if weights is None:
            weights = self.weights  # Use the class variable if no weights are provided

        # Simulate portfolio returns using the covariance matrix
        port_returns = np.dot(self.returns, weights)

        # Calculate VaR
        var = np.percentile(port_returns, 100 * alpha)

        # Calculate CVaR (average of returns that are worse than VaR)
        cvar = np.abs(port_returns[port_returns <= var].mean())
        return cvar


    def portfolio_shape_VaR(self, alpha=0.05, weights=None):
        # Calculate the portfolio returns
        if weights is None:
            weights = self.weights  # Use the class variable if no weights are provided

        # Simulate portfolio returns using the covariance matrix
        port_returns = np.dot(self.returns, weights)
        mean_return = self.calculate_mean_returns(weights)

        var = np.abs(np.percentile(port_returns, 100 * alpha))
        shape_VaR = (mean_return - self.risk_free_rate)/var
        return shape_VaR


    def portfolio_shape_CVaR(self, alpha=0.05, weights=None):
        # Calculate the portfolio returns
        if weights is None:
            weights = self.weights  # Use the class variable if no weights are provided

        # Simulate portfolio returns using the covariance matrix
        port_returns = np.dot(self.returns, weights)
        mean_return = self.calculate_mean_returns(weights)
        # Calculate VaR
        var = np.percentile(port_returns, 100 * alpha)

        # Calculate CVaR (average of returns that are worse than VaR)
        cvar = np.abs(port_returns[port_returns <= var].mean())
        shape_CVaR = (mean_return - self.risk_free_rate)/cvar
        return shape_CVaR


    def portfolio_performance(self, alpha = 0.05, weights=None):
        # Returns the annualized portfolio return and standard deviation
        mean_return = self.calculate_mean_returns(weights)
        stdev = self._calculate_stdev(weights)
        shape = self.portfolio_shape(weights)
        vaR = self.portfolio_VaR(alpha = alpha, weights = weights)
        cvaR = self.portfolio_CVaR(alpha = alpha, weights = weights)
        return mean_return, stdev, shape, vaR, cvaR


if __name__ == "__main__":
    # Example Usage
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JNJ", "PFE", "VZ", "T", "KO", "SPY", "META", "NVDA", "JD", "CORT", "PRA"]
    start_date = "2013-12-31"
    end_date = "2021-01-01"
    dfport = yf.download(tickers, start_date, end_date)
        
    date_filter_start = "2019-01-01"
    date_filter_end = "2023-12-31"
    tickers1 = ["JD", "AAPL", "META", "NVDA", "AMZN"]
    weigths = [1/len(tickers1)]*len(tickers1)
    portfolio = Portfolio(data_in = dfport, tickers = tickers1, weights = weigths, start = date_filter_start, end = date_filter_end, risk_free_rate = 0.01)

    print(portfolio.returns.corr())
    #annual_return, annual_stdev, shape = portfolio.portfolio_performance()
    alpha, beta =  portfolio.calculate_alpha_beta("^GSPC")

    print(f"Annual Portfolio Return: {annual_return:.2%}")
    print(f"Annual Portfolio Stdev: {annual_stdev:.2%}")
    print(f"alpha: {alpha: 3f}")
    print(f"beta: {beta: 3f}")