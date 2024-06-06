from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone
import numpy as np

class CustomGradientBoostingClassifier:
    def __init__(self, base_estimator=None, n_estimators=100, learning_rate=0.1):
        self.base_estimator = base_estimator if base_estimator is not None else DecisionTreeRegressor(max_depth=3)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        self.initial_prediction = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        assert len(self.classes_) == 2, "This implementation supports only binary classification"

        # Transform y to {-1, 1}
        y_transformed = np.where(y == self.classes_[0], -1, 1)
        
        # Initial prediction: log(odds)
        self.initial_prediction = np.log((y_transformed == 1).sum() / (y_transformed == -1).sum())
        f_m = np.full(y.shape, self.initial_prediction)
        
        for _ in range(self.n_estimators):
            # transforming into logistic function
            p_m = 1 / (1 + np.exp(-f_m)) 
            residuals = y_transformed - p_m
            estimator = clone(self.base_estimator)
            estimator.fit(X, residuals)
            self.estimators.append(estimator)
            f_m += self.learning_rate * estimator.predict(X)

    def predict_proba(self, X):
        f_m = np.full(X.shape[0], self.initial_prediction)
        for estimator in self.estimators:
            f_m += self.learning_rate * estimator.predict(X)
        # logistic f again - into probability
        p_m = 1 / (1 + np.exp(-f_m))
        return np.vstack((1 - p_m, p_m)).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.where(proba[:, 1] > 0.5, self.classes_[1], self.classes_[0])