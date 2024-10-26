import numpy as np
import pandas as pd
from scipy.optimize import minimize

class Optimize:
    def __init__(self, portfolio, alpha_beta_calculator):
        self.portfolio = portfolio
        self.alpha_beta_calculator = alpha_beta_calculator  # Instance of AlphaBeta class

    def _optimize_weights(self, objective_function, constraints=None):
        # Define bounds for weights (0 to 1 for each asset)
        bounds = tuple((0, 1) for _ in range(len(self.portfolio.tickers)))

        # Initial guess for weights (equal distribution)
        initial_weights = np.array(self.portfolio.weights)

        # Run the optimization
        result = minimize(objective_function, initial_weights, bounds=bounds, constraints=constraints)

        if result.success:
            return result.x  # Return optimized weights
        else:
            raise ValueError("Optimization failed: " + result.message)

    def largest_alpha(self):
        # Objective function to maximize alpha
        def objective(weights):
            # Update portfolio weights
            self.portfolio.weights = weights
            alpha, _ = self.alpha_beta_calculator.calculate_alpha_beta(self.portfolio)
            return -alpha  # Minimize negative alpha (maximize alpha)

        return self._optimize_weights(objective)

    def smallest_beta(self):
        # Objective function to minimize beta
        def objective(weights):
            self.portfolio.weights = weights
            _, beta = self.alpha_beta_calculator.calculate_alpha_beta(self.portfolio)
            return beta  # Minimize beta

        return self._optimize_weights(objective)

    def minimize_beta_given_alpha(self, target_alpha):
        # Objective function to minimize beta with a constraint on alpha
        def objective(weights):
            self.portfolio.weights = weights
            _, beta = self.alpha_beta_calculator.calculate_alpha_beta(self.portfolio)
            return beta  # Minimize beta

        # Constraint to ensure the portfolio alpha equals target_alpha
        def constraint(weights):
            self.portfolio.weights = weights
            alpha, _ = self.alpha_beta_calculator.calculate_alpha_beta(self.portfolio)
            return alpha - target_alpha  # Should be zero

        constraints = {'type': 'eq', 'fun': constraint}
        return self._optimize_weights(objective, constraints)

    def minimize_alpha_given_beta(self, target_beta):
        # Objective function to minimize alpha with a constraint on beta
        def objective(weights):
            self.portfolio.weights = weights
            alpha, _ = self.alpha_beta_calculator.calculate_alpha_beta(self.portfolio)
            return alpha  # Minimize alpha

        # Constraint to ensure the portfolio beta equals target_beta
        def constraint(weights):
            self.portfolio.weights = weights
            _, beta = self.alpha_beta_calculator.calculate_alpha_beta(self.portfolio)
            return beta - target_beta  # Should be zero

        constraints = {'type': 'eq', 'fun': constraint}
        return self._optimize_weights(objective, constraints)

    def find_weights_for_alpha_and_beta(self, target_alpha, target_beta):
        # Objective function to minimize the sum of squared differences from target alpha and beta
        def objective(weights):
            self.portfolio.weights = weights
            alpha, beta = self.alpha_beta_calculator.calculate_alpha_beta(self.portfolio)
            return (alpha - target_alpha) ** 2 + (beta - target_beta) ** 2  # Minimize the difference

        return self._optimize_weights(objective)

# Example Usage
if __name__ == "__main__":
    # Assuming Portfolio and AlphaBeta classes are defined as in previous examples
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    weights = [0.3, 0.2, 0.3, 0.2]
    start_date = "2013-12-31"
    end_date = "2021-01-01"

    portfolio = Portfolio(tickers, weights, start_date, end_date)
    alphabeta = AlphaBeta()
    
    optimizer = Optimize(portfolio, alphabeta)

    # Example of finding weights for the largest alpha
    largest_alpha_weights = optimizer.largest_alpha()
    print("Weights for largest alpha:", largest_alpha_weights)

    # Example of finding weights for the smallest beta
    smallest_beta_weights = optimizer.smallest_beta()
    print("Weights for smallest beta:", smallest_beta_weights)

    # Example of minimizing beta given a target alpha
    target_alpha = 0.05  # Example target alpha
    weights_for_min_beta = optimizer.minimize_beta_given_alpha(target_alpha)
    print("Weights minimizing beta for target alpha:", weights_for_min_beta)

    # Example of minimizing alpha given a target beta
    target_beta = 1.0  # Example target beta
    weights_for_min_alpha = optimizer.minimize_alpha_given_beta(target_beta)
    print("Weights minimizing alpha for target beta:", weights_for_min_alpha)

    # Example of finding weights for given alpha and beta
    target_alpha = 0.05  # Example target alpha
    target_beta = 1.0    # Example target beta
    weights_for_alpha_beta = optimizer.find_weights_for_alpha_and_beta(target_alpha, target_beta)
    print("Weights for given alpha and beta:", weights_for_alpha_beta)