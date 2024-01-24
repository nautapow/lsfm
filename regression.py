import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import make_scorer
from sklearn.datasets import make_regression
from sklearn.metrics import make_scorer, r2_score
import pandas as pd

def regression_test(x, y):
    X = np.array(x).reshape(-1,1)
    y = np.array(y)
    
    def calculate_metrics(model, X, y, num_params):
        y_pred = model.predict(X)
        residuals = y - y_pred
        mse = np.mean(residuals**2)
        aic = len(y) * np.log(mse) + 2 * num_params
        bic = len(y) * np.log(mse) + num_params * np.log(len(y))
        
        
        return aic, bic
    
    # Try polynomial degrees from 1 to 6
    degrees = np.arange(1, 7)
    
    # Perform polynomial regression for each degree and calculate AIC, BIC, and R-squared with LOOCV
    aic_scores = []
    bic_scores = []
    r2_scores = []
    ytests = []
    ypreds = []
    
    for degree in degrees:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        
        # Use Leave-One-Out Cross-Validation
        loocv = LeaveOneOut()
        
        # Calculate AIC, BIC, and R-squared scores using LOOCV
        scores = []
        for train_index, test_index in loocv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model.fit(X_train, y_train)
            scores.append(calculate_metrics(model, X_test, y_test, degree + 1))
            
            y_pred = model.predict(X_test)
            ytests += list(y_test)
            ypreds += list(y_pred)
        
        rr = r2_score(ytests, ypreds)
        scores_mean = np.mean(scores, axis=0)
        aic_scores.append(scores_mean[0])
        bic_scores.append(scores_mean[1])
        r2_scores.append(rr)
    
    # Plot the results
    plt.plot(degrees, aic_scores, label='AIC')
    plt.plot(degrees, bic_scores, label='BIC')
    plt.plot(degrees, r2_scores, label='R-squared')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Score')
    plt.title('AIC, BIC, and R-squared for Polynomial Regression with LOOCV')
    plt.legend()
    plt.show()
    
    return aic_scores, bic_scores, r2_scores
