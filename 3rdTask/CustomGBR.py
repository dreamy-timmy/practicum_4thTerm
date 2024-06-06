import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor

class CustomGradientBoostingRegressor:
    def __init__(self, base_estimator=None, n_estimators=100, learning_rate=0.1):
        self.base_estimator = base_estimator if base_estimator is not None else DecisionTreeRegressor(max_depth=3)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        self.initial_prediction = None

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        residuals = y - self.initial_prediction

        for _ in range(self.n_estimators):
            estimator = clone(self.base_estimator)
            estimator.fit(X, residuals)
            predictions = estimator.predict(X)
            residuals -= self.learning_rate * predictions
            self.estimators.append(estimator)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.initial_prediction)
        for estimator in self.estimators:
            y_pred += self.learning_rate * estimator.predict(X)
        return y_pred